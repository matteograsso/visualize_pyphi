import pyphi
import itertools

from pyphi import relations as rels
from pyphi.models.subsystem import FlatCauseEffectStructure as sep
from itertools import product
import numpy as np
from tqdm.auto import tqdm
import random
from operator import attrgetter
from joblib import Parallel, delayed


CAUSE = pyphi.direction.Direction(0)
EFFECT = pyphi.direction.Direction(1)


def specified_elements(ces, direction):
    """Return elements specified by at least one purview.

    Args:
        ces (pyphi.models.subsystem.FlatCauseEffectStructure): The distinctions to check.

    Returns:
        set: The elements.
    """
    return set(
                [
                    u
                    for distinction in ces
                    for u in distinction.purview
                    if (distinction.direction == direction
                    and not (len(distinction.mechanism) == 1 and distinction.mechanism == distinction.purview))
                ]
    )


def is_trivially_reducible(subsystem, ces):
    """Return whether a system is trivially reducible.

    Checks that all elements are specified by a purview in both directions.

    Args:
        subsystem (pyphi.subsystem.Subsystem): The subsystem of interest.
        ces (pyphi.models.subsystem.FlatCauseEffectStructure): The distinctions
            to check for reducibility.

    Returns:
        bool: True if the subsystem is trivially reducible, False otherwise
    """
    return any(
        element not in specified_elements(ces, CAUSE)
        or element not in specified_elements(ces, EFFECT)
        for element in subsystem.node_indices
    )


def get_maximal_ces(system, ces=None, max_k=3, compositional_states=[], relations=[]):
    # Find the maximally irreducible CES for a given subsystem

    # unfold unfiltered, separated ces for the system
    if ces == None:
        directions = [pyphi.direction.Direction.CAUSE, pyphi.direction.Direction.EFFECT]
        print("unfolding CES")
        ces = pyphi.models.subsystem.CauseEffectStructure(
            [
                system.find_mice(d, m)
                for d in directions
                for m in pyphi.utils.powerset(system.node_indices, nonempty=True)
            ],
            subsystem=system,
        )
        for m in ces:
            m.node_labels = system.node_labels
    ces = [m for m in ces if m.phi]

    # Find all compositional states, if they are not provided
    if len(compositional_states) == 0:
        compositional_states = get_all_compositional_states(ces)

    # Filter the CES and compute Big Phi for every compositional state
    big_phi = 0
    maximal = []

    for compositional_state in (
        compositional_states
        if len(compositional_states) < 2
        else tqdm(
            compositional_states, desc="Computing Big Phi for all compositional states",
        )
    ):
        # Filter the CES and relations
        (
            filtered_ces,
            filtered_relations,
            compositional_state,
        ) = compute_rels_and_ces_for_compositional_state(
            system, compositional_state, ces, max_k, relations
        )

        if not filtered_ces:
            phi = 0
            cut = ()
        else:
            # Compute Big Phi
            phi, cut = get_big_phi(
                system, filtered_ces, filtered_relations, system.node_indices
            )

        # Save values for highest BigPhi So far (or append if a tie)
        if phi > big_phi:
            maximal = [
                {
                    "ces": filtered_ces,
                    "relations": filtered_relations,
                    "compositional_state": compositional_state,
                    "big phi": phi,
                    "MIP": cut,
                }
            ]
            big_phi = phi
        elif phi == big_phi and phi > 0:
            maximal.append(
                {
                    "ces": filtered_ces,
                    "relations": filtered_relations,
                    "compositional_state": compositional_state,
                    "big phi": phi,
                    "MIP": cut,
                }
            )
        elif phi == big_phi and phi == 0:
            maximal = [
                {
                    "ces": filtered_ces,
                    "relations": filtered_relations,
                    "compositional_state": compositional_state,
                    "big phi": phi,
                    "MIP": cut,
                }
            ]

    return maximal


def compute_rels_and_ces_for_compositional_state(
    system, state, ces, max_k=3, relations=[]
):

    if type(state) is tuple:
        state = compositional_state_from_system_state(state)

    # Filter the distinctions
    filtered_ces = filter_ces_by_compositional_state(ces, state)

    if is_trivially_reducible(system, filtered_ces):
        return ((), (), ())

    else:
        # Filter the relations
        if len(relations) > 0:
            filtered_relations = filter_relations(relations, filtered_ces)
        else:
            filtered_relations = compute_relations(system, filtered_ces, max_k=max_k)

        return (
            filtered_ces,
            filtered_relations,
            state,
        )


def filter_ces(
    subsystem,
    ces,
    compositional_state,
    relations=[],
    max_relations_k=3,
    n_jobs=120,
    parallel=True,
    verbose=5,
    batch_size=1000,
):
    # Check for trivial reducibility
    if is_trivially_reducible(subsystem, ces):
        print("Subsystem is trivially reducible!")

    else:
        if compositional_state == None:
            purview_states = dict()  # compositional_state.copy()
            for mice in ces:
                if not (mice.direction, mice.purview) in purview_states.keys():
                    purview_states[(mice.direction, mice.purview)] = [mice]
                else:
                    purview_states[(mice.direction, mice.purview)].append(mice)

            all_cess = list(itertools.product(*purview_states.values()))
        else:
            # next we run through all the mices and append any mice that has a state corresponding to the compositional state
            mices_with_correct_state = dict()  # compositional_state.copy()
            for mice in ces:
                if (
                    tuple(mice.specified_state[0])
                    == compositional_state[mice.direction][mice.purview]
                ):
                    if (
                        not (mice.direction, mice.purview)
                        in mices_with_correct_state.keys()
                    ):
                        mices_with_correct_state[(mice.direction, mice.purview)] = [
                            mice
                        ]
                    else:
                        mices_with_correct_state[(mice.direction, mice.purview)].append(
                            mice
                        )

            all_cess = list(itertools.product(*mices_with_correct_state.values()))

        max_ces = []
        if parallel and len(all_cess) > 20:
            max_ces = Parallel(
                n_jobs=n_jobs, verbose=verbose, backend="loky", batch_size=batch_size
            )(
                delayed(resolve_conflicts)(subsystem, ces, max_relations_k, relations)
                for ces in tqdm(all_cess)
            )
        else:
            max_ces = [
                resolve_conflicts(subsystem, ces, max_relations_k, relations)
                for ces in tqdm(all_cess)
            ]

        return max(max_ces, key=lambda c: c[0]["big phi"])


def resolve_conflicts(subsystem, ces, max_k=3, relations=[]):

    # the following two loops do the filtering, and are identical except the first does cause and the other effect
    # If the same purview is specified by multiple mechinisms, we only keep the one with max phi
    causes = [mice for mice in ces if mice.direction == CAUSE]
    effects = [mice for mice in ces if mice.direction == EFFECT]

    # remove any unlinked mice
    causeeffect_mechanisms = set([cause.mechanism for cause in causes]).intersection(
        set([effect.mechanism for effect in effects])
    )
    causeeffects = causes + effects
    filtered_ces = [
        mice for mice in causeeffects if mice.mechanism in causeeffect_mechanisms
    ]
    return get_maximal_ces(
        subsystem,
        ces=pyphi.models.CauseEffectStructure(filtered_ces),
        max_k=max_k,
        relations=relations,
    )


def get_big_phi(subsystem, ces, relations, indices, partitions=None):

    if is_trivially_reducible(subsystem, ces):
        return 0, ()
    else:
        # Getting the small phi values of each (linked) distinctions and computes their sum
        # NOTE: this first part was post hoc added to make computations work with linked distinctions
        phis = [
            min([cause.phi, effect.phi])
            for cause, effect in zip(
                [mice for mice in ces if mice.direction == CAUSE],
                [mice for mice in ces if mice.direction == EFFECT],
            )
        ]
        sum_of_small_phi = sum(phis) + sum([r.phi for r in relations])

        # Assuming single unit systems are irreducible by definition
        if len(indices) == 1:
            return sum_of_small_phi, (((), ()), "disintegration")

        # Getting all possible bipartitions of the system (if no specific set of partitions were provided)
        if partitions == None:
            partitions = [
                part
                for part in pyphi.partition.bipartition(indices)
                if all([len(p) > 0 for p in part])
            ]

        # keeping track of the informativeness for each cut, to find the minimal one to use in Big Phi
        informativeness = np.inf

        # Looping through every bipartition
        for parts in (
            tqdm(partitions, desc="System partitions")
            if len(partitions) > 100 and len(indices) > 4
            else partitions
        ):
            # looping through the four types of cuts between the parts
            for p1, p2, direction in product(parts, parts, [CAUSE, EFFECT]):
                # Making sure the two parts are different
                if not p1 == p2:

                    # Finding the mices that are untouched by the cut
                    untouched_mices = pyphi.models.CauseEffectStructure(
                        [
                            mice
                            for mice in ces
                            if not distinction_touched(mice, p1, p2, direction)
                        ]
                    )

                    # Getting the mechanisms that specify untouched relations
                    # NOTE: This is needed do to the linking of purviews to kill purviews linked to killed "distinctions"
                    causeeffect_mechanisms = set(
                        [
                            mice.mechanism
                            for mice in untouched_mices
                            if mice.direction == CAUSE
                        ]
                    ).intersection(
                        set(
                            [
                                mice.mechanism
                                for mice in untouched_mices
                                if mice.direction == EFFECT
                            ]
                        )
                    )

                    # Keeping MICEs specified by mechanisms that have both a cause and an effect
                    untouched_ces = pyphi.models.CauseEffectStructure(
                        [
                            mice
                            for mice in untouched_mices
                            if mice.mechanism in causeeffect_mechanisms
                        ]
                    )

                    # Computing relations for the untouched CES
                    untouched_relations = [
                        r for r in relations if relation_untouched(untouched_ces, r)
                    ]

                    # Getting small phi for distinctions
                    # NOTE: Assuming the causes and effects are ordered in the same way...
                    phis = [
                        min([cause.phi, effect.phi])
                        for cause, effect in zip(
                            [mice for mice in untouched_ces if mice.direction == CAUSE],
                            [
                                mice
                                for mice in untouched_ces
                                if mice.direction == EFFECT
                            ],
                        )
                    ]

                    # computing the sum of small phi for Big Phi computation
                    sum_phi_untouched = sum(phis) + sum(
                        [r.phi for r in untouched_relations]
                    )

                    # Checking how much small phi is lost by the partition
                    lost_phi = sum_of_small_phi - sum_phi_untouched

                    # If this cut is the least destructive cut found so far, we save some values
                    if lost_phi < informativeness:
                        informativeness = lost_phi
                        min_cut = parts, p1, p2, direction

        # Computing selectivity
        # NOTE: the denominator must be corrected to allow comparison of systems with differing sizes
        selectivity = sum_of_small_phi / 2 ** len(indices)

        # Compute Big Phi and return
        big_phi = selectivity * (informativeness)
        return big_phi, min_cut


def compute_relations(subsystem, ces, max_k=3, num_relations=False):
    # Compute a number of relations up to order "max_k"

    relations = []
    # Loop through every order relation between 2 and max_k
    for k in range(2, max_k + 1):
        # find all the relata
        relata = [
            rels.Relata(subsystem, mices)
            for mices in itertools.combinations(ces, k)
            if all([mice.phi > 0 for mice in mices])
        ]

        # Pick a random sample of the relata, if only some of them are to be checked for irreducibility
        if num_relations:
            relata = random.sample(relata, num_relations)

        # Compute the relations for all the relata
        k_relations = [
            rels.relation(relatum)
            for relatum in (tqdm(relata) if len(relata) > 5000 else relata)
        ]

        # remove any reducible relations
        k_relations = list(filter(lambda r: r.phi > 0, k_relations))
        relations.extend(k_relations)
    # ces = [m for m in ces if m.phi]
    # relations = list(
    #     pyphi.relations.relations(
    #         subsystem,
    #         ces,
    #         min_order=2,
    #         max_order=max_k,
    #         parallel=True,
    #         # parallel_kwargs=dict(n_jobs=120, batch_size=10000, verbose=0),
    #     )
    # )

    return relations


def distinction_touched(mice, part1, part2, direction):
    # Check if a particular MICE is touched by a specific partition
    mechanism_in_part1 = any([m in part1 for m in mice.mechanism])
    purview_in_part2 = any([p in part2 for p in mice.purview])
    correct_direction = direction == mice.direction

    return mechanism_in_part1 and purview_in_part2 and correct_direction


def relation_untouched(untouched_ces, relation):
    # Check if a relation is specified by an "untouched" CES
    relata_in_ces = all([relatum in untouched_ces for relatum in relation.relata])

    return relata_in_ces


def unfold_separated_ces(system):
    CAUSE = pyphi.direction.Direction(0)
    EFFECT = pyphi.direction.Direction(1)
    purviews = tuple(pyphi.utils.powerset(system.node_indices, nonempty=True))

    mices = [
        system.find_mice(direction, mechanism, purviews=purviews)
        for mechanism in pyphi.utils.powerset(system.node_indices, nonempty=True)
        for direction in [CAUSE, EFFECT]
    ]

    # remove any mice with phi=0
    mices = [mice for mice in mices if mice.phi > 0]

    # kill unlinked mices
    causeeffect_mechanisms = set(
        [mice.mechanism for mice in mices if mice.direction == CAUSE]
    ).intersection(set([mice.mechanism for mice in mices if mice.direction == EFFECT]))
    filtered_ces = [mice for mice in mices if mice.mechanism in causeeffect_mechanisms]

    ces = pyphi.models.CauseEffectStructure(mices)
    for m in ces:
        m.node_labels = system.node_labels
    return ces


def compositional_state_from_system_state(state, system_indices=None):

    if system_indices == None:
        system_indices = tuple(range(len(state)))

    # if a single state is sent in, it assumes thae cause and effect states are the same
    if type(state) is tuple:
        cause_state = state
        effect_state = state
    else:
        cause_state = state[0]
        effect_state = state[1]

    # returns a dictionary with cause and effect compositional states as sub-dictionaries
    # the subdictionaries have purview indices as keys and their state as value
    return {
        direction: {
            subset_elements: tuple(state[i] for i in subset_elements)
            for subset_elements in pyphi.utils.powerset(system_indices, nonempty=True)
        }
        for state, direction in zip(
            [cause_state, effect_state],
            [pyphi.direction.Direction.CAUSE, pyphi.direction.Direction.EFFECT],
        )
    }


def get_all_compositional_states(ces):
    c = pyphi.direction.Direction.CAUSE
    e = pyphi.direction.Direction.EFFECT

    # now we make a nested dict that contains all the purviews with their related states, mechanisms, and phi values
    all_purviews = {c: dict(), e: dict()}

    # we loop through every mice in the separated CES
    for mice in ces:

        # define some variables for later use
        purview = mice.purview
        mechanism = mice.mechanism
        max_state = tuple(mice.specified_state[0])
        direction = mice.direction
        phi = mice.phi

        # check if the purview is already in our dict
        if len(max_state) > 0:
            if purview in all_purviews[direction]:
                # check if the purview is already represented with this state
                if not max_state in all_purviews[direction][purview]:
                    all_purviews[direction][purview].append(max_state)
            else:
                all_purviews[direction][purview] = [max_state]

    cause_states = [
        {
            purv: purview_state
            for purv, purview_state in zip(all_purviews[c].keys(), purview_states)
        }
        for purview_states in product(*all_purviews[c].values())
    ]
    effect_states = [
        {
            purv: purview_state
            for purv, purview_state in zip(all_purviews[e].keys(), purview_states)
        }
        for purview_states in product(*all_purviews[e].values())
    ]

    all_compositional_states = [
        {c: c_state, e: e_state}
        for c_state, e_state in product(cause_states, effect_states,)
    ]
    return all_compositional_states


def filter_ces_by_compositional_state(ces, compositional_state):

    # first separate the ces into mices and define the directions
    c = pyphi.direction.Direction.CAUSE
    e = pyphi.direction.Direction.EFFECT

    # next we run through all the mices and append any mice that has a state corresponding to the compositional state
    mices_with_correct_state = [
        mice
        for mice in ces
        if (
            tuple(mice.specified_state[0])
            == compositional_state[mice.direction][mice.purview]
        )
    ]

    # next we find the set of purviews (congruent with the compositional state) specified by the system
    cause_purviews = set(
        [mice.purview for mice in mices_with_correct_state if mice.direction == c]
    )
    effect_purviews = set(
        [mice.purview for mice in mices_with_correct_state if mice.direction == e]
    )

    # the following two loops do the filtering, and are identical except the first does cause and the other effect
    # If the same purview is specified by multiple mechinisms, we only keep the one with max phi
    causes = []
    for purview in cause_purviews:
        mices = list(
            filter(
                lambda mice: mice.direction == c and mice.purview == purview,
                mices_with_correct_state,
            )
        )
        causes.append(mices[np.argmax([mice.phi for mice in mices])])

    effects = []
    for purview in effect_purviews:
        mices = list(
            filter(
                lambda mice: mice.direction == e and mice.purview == purview,
                mices_with_correct_state,
            )
        )
        effects.append(mices[np.argmax([mice.phi for mice in mices])])

    # remove any unlinked mice
    causeeffect_mechanisms = set([cause.mechanism for cause in causes]).intersection(
        set([effect.mechanism for effect in effects])
    )
    causeeffects = causes + effects
    filtered_ces = [
        mice for mice in causeeffects if mice.mechanism in causeeffect_mechanisms
    ]

    return pyphi.models.CauseEffectStructure(filtered_ces)


def filter_relations(relations, filtered_ces):
    return list(
        filter(
            lambda r: all([relatum in filtered_ces for relatum in r.relata]), relations
        )
    )


def filter_using_sum_of_distinction_phi(ces, relations):

    # get all purviews
    all_purviews = set([(mice.purview, mice.direction) for mice in ces])

    # keep only the mice with max smallphi among conflicts (pick first if ties---it doesnt matter)
    filtered_ces = [
        max(
            [mice for mice in ces if (mice.purview, mice.direction) == purview],
            key=attrgetter("phi"),
        )
        for purview in all_purviews
    ]

    compositional_state = get_all_compositional_states(filtered_ces)
    filtered_relations = filter_relations(relations, filtered_ces)
    return filtered_ces, filtered_relations, compositional_state


def filter_using_sum_of_phi(ces, relations, all_compositional_states):
    phis = {
        i: sum(
            [
                sum([c.phi for c in filter_ces_by_compositional_state(ces, comp[1])]),
                sum(
                    [
                        r.phi
                        for r in filter_relations(
                            relations, filter_ces_by_compositional_state(ces, comp[1])
                        )
                    ]
                ),
            ]
        )
        for i, comp in (
            enumerate(
                tqdm(all_compositional_states)
                if len(all_compositional_states) > 100
                else enumerate(all_compositional_states)
            )
        )
    }

    max_compositional_state = all_compositional_states[
        list(phis.values()).index(max(phis.values()))
    ]
    filtered_ces = filter_ces_by_compositional_state(ces, max_compositional_state)
    filtered_relations = filter_relations(relations, filtered_ces)
    return filtered_ces, filtered_relations, max_compositional_state


def get_filtered_ces(ces, state, system=None):

    if state == "actual":
        state = tuple(system.state[i] for i in system.node_indices)
        state = compositional_state_from_system_state(state)
        filtered_ces = filter_ces_by_compositional_state(ces, state)

    elif type(state) is tuple or type(state) is list:
        state = compositional_state_from_system_state(state)
        filtered_ces = filter_ces_by_compositional_state(ces, state)

    elif type(state) is dict:
        filtered_ces = filter_ces_by_compositional_state(ces, state)
        filtered_rels = filter_relations(relations, filtered_ces)

    elif state == "max_phi_distinction":
        filtered_ces, filtered_rels, state = filter_using_sum_of_distinction_phi(
            ces, []
        )

    elif state == "max_phi":
        compositional_states = get_all_compositional_states(ces)
        filtered_ces, filtered_rels, state = filter_using_sum_of_phi(
            ces, [], compositional_states
        )

    elif state == "BIG_PHI":
        maximal = get_maximal_ces(system, ces=ces, max_k=3)
        filtered_ces = maximal["ces"]
        state = maximal["compositional_state"]

    return (
        filtered_ces,
        state,
    )


def get_filtered_ces_and_rels(ces, relations, state, system=None):
    if state == "actual":
        state = tuple(system.state[i] for i in system.node_indices)
        state = compositional_state_from_system_state(state)
        filtered_ces = filter_ces_by_compositional_state(ces, state)
        filtered_rels = filter_relations(relations, filtered_ces)

    elif type(state) is tuple or type(state) is list:
        state = compositional_state_from_system_state(state)
        filtered_ces = filter_ces_by_compositional_state(ces, state)
        filtered_rels = filter_relations(relations, filtered_ces)

    elif type(state) is dict:
        filtered_ces = filter_ces_by_compositional_state(ces, state)
        filtered_rels = filter_relations(relations, filtered_ces)

    elif state == "max_phi_distinction":
        filtered_ces, filtered_rels, state = filter_using_sum_of_distinction_phi(
            ces, relations
        )

    elif state == "max_phi":
        compositional_states = get_all_compositional_states(ces)
        filtered_ces, filtered_rels, state = filter_using_sum_of_phi(
            ces, relations, compositional_states
        )

    elif state == "BIG_PHI":
        maximal = get_maximal_ces(system, ces=ces, max_k=3)
        filtered_ces = maximal["ces"]
        filtered_rels = maximal["relations"]
        state = maximal["compositional_state"]

    return (
        filtered_ces,
        filtered_rels,
        state,
    )


def get_all_ces(system, ces=None, max_k=3, relations=[]):

    # unfold ces
    if ces == None:
        directions = [pyphi.direction.Direction.CAUSE, pyphi.direction.Direction.EFFECT]
        print("unfolding CES")
        ces = pyphi.models.subsystem.CauseEffectStructure(
            [
                system.find_mice(d, m)
                for d in directions
                for m in pyphi.utils.powerset(system.node_indices, nonempty=True)
            ],
            subsystem=system,
        )
        for m in ces:
            m.node_labels = system.node_labels

    all_ces = []
    compositional_states = get_all_compositional_states(ces)
    big_phi = 0
    for compositional_state in tqdm(compositional_states):
        (
            filtered_ces,
            filtered_relations,
            compositional_state,
        ) = compute_rels_and_ces_for_compositional_state(
            system, compositional_state, ces, max_k, relations
        )
        phi, cut = get_big_phi(
            system, filtered_ces, filtered_relations, system.node_indices
        )

        all_ces.append(
            {
                "ces": filtered_ces,
                "relations": filtered_relations,
                "compositional_state": compositional_state,
                "big phi": phi,
            }
        )

    return all_ces


def get_untouced_ces_and_rels(ces, relations, parts):
    untouched_ces = pyphi.models.CauseEffectStructure(
        [
            mice
            for mice in ces
            if not distinction_touched(mice, parts[1], parts[2], parts[3])
        ]
    )
    untouched_relations = [r for r in relations if relation_untouched(untouched_ces, r)]

    return untouched_ces, untouched_relations


def get_component_phi(ces, relations, component_distinctions):

    component_ces = [mice for mice, d in zip(ces, component_distinctions) if d]
    component_relations = [r for r in relations if relation_untouched(component_ces, r)]

    sum_of_small_phi = sum([mice.phi for mice in component_ces]) + sum(
        [r.phi for r in component_relations]
    )

    min_phi = np.inf
    dominant_distinction = ()
    component_phi = 0
    for i, untouched_ces in enumerate(
        [
            pyphi.models.CauseEffectStructure(
                component_ces[:i] + component_ces[i + 1 :]
            )
            for i in range(len(component_ces) - 1)
        ]
    ):
        untouched_relations = [
            r for r in component_relations if relation_untouched(untouched_ces, r)
        ]

        sum_phi_untouched = (
            sum([mice.phi for mice in untouched_ces])
            + sum([r.phi for r in untouched_relations])
            if len(untouched_relations) > 0
            else sum([mice.phi for mice in untouched_ces])
        )

        max_possible_phi_untouched = sum(
            [len(mice.mechanism) for mice in untouched_ces]
        ) + sum(
            [
                min([len(relatum.purview) for relatum in relation.relata])
                for relation in untouched_relations
            ]
        )

        lost_phi = sum_of_small_phi - sum_phi_untouched

        if lost_phi < min_phi:
            min_phi = lost_phi
            dominant_distinction = component_ces[i]
            component_phi = sum_phi_untouched / max_possible_phi_untouched * lost_phi

    return component_phi, dominant_distinction


def add_missing_purviews(ces, filtered_ces):

    all_purviews = set([(mice.purview, mice.direction) for mice in ces])
    filtered_purviews = set([(mice.purview, mice.direction) for mice in filtered_ces])

    filtered_ces = list(filtered_ces)

    for purview, direction in all_purviews.difference(filtered_purviews):
        phi = 0
        missing_mice = None
        for mice in ces:
            if (
                mice.purview == purview
                and mice.direction == direction
                and mice.phi > phi
            ):
                missing_mice = mice

        filtered_ces.append(missing_mice)

    return pyphi.models.CauseEffectStructure(filtered_ces)


def context_relations(relations, distinctions):
    return [
        relation
        for relation in relations
        if (
            any([relatum in distinctions for relatum in relation.relata])
            and not all([relatum in distinctions for relatum in relation.relata])
        )
    ]


def distinction_touched_old(mice, part1, part2, direction):
    mechanism_cut = all(
        [any([m in part for m in mice.mechanism]) for part in [part1, part2]]
    )
    purview_cut = all(
        [any([p in part for p in mice.purview]) for part in [part1, part2]]
    )
    connection_cut = (
        all([m in part1 for m in mice.mechanism])
        and all([p in part2 for p in mice.purview])
        and mice.direction == direction
    )
    return mechanism_cut or purview_cut or connection_cut


### PHI FOLD CODE
def get_node_ixs(ces):
    return tuple(
        set(
            [u for mice in ces for u in mice.mechanism]
            + [u for mice in ces for u in mice.purview]
        )
    )


def get_mechanism_subtext(ces, mechanism):
    mechanisms = list(pyphi.utils.powerset(mechanism))
    return pyphi.models.CauseEffectStructure(
        [mice for mice in ces if mice.mechanism in mechanisms]
    )


def get_mechanism_supertext(ces, mechanism):
    node_indices = get_node_ixs(ces)
    mechanisms = [
        ixs
        for ixs in pyphi.utils.powerset(node_indices)
        if all([ix in ixs for ix in mechanism])
    ]
    return pyphi.models.CauseEffectStructure(
        [mice for mice in ces if mice.mechanism in mechanisms]
    )


def get_mechanism_paratext(ces, mechanism):
    node_indices = get_node_ixs(ces)
    mechanisms = [
        ixs
        for ixs in pyphi.utils.powerset(node_indices)
        if any([ix in ixs for ix in mechanism])
        and not all([ix in ixs for ix in mechanism])
    ]
    return pyphi.models.CauseEffectStructure(
        [mice for mice in ces if mice.mechanism in mechanisms]
    )


def get_mechanism_context(ces, mechanism):
    return pyphi.models.CauseEffectStructure(
        set(
            [mice for mice in get_mechanism_subtext(ces, mechanism)]
            + [mice for mice in get_mechanism_supertext(ces, mechanism)]
            + [mice for mice in get_mechanism_paratext(ces, mechanism)]
        )
    )


def get_purview_subtext(ces, purview):
    purviews = list(pyphi.utils.powerset(purview))
    return pyphi.models.CauseEffectStructure(
        [mice for mice in ces if mice.purview in purviews]
    )


def get_purview_supertext(ces, purview):
    node_indices = get_node_ixs(ces)
    purviews = [
        ixs
        for ixs in pyphi.utils.powerset(node_indices)
        if all([ix in purview for ix in ixs])
    ]
    return pyphi.models.CauseEffectStructure(
        [mice for mice in ces if mice.purview in purviews]
    )


def get_purview_paratext(ces, purview):
    node_indices = get_node_ixs(ces)
    purviews = [
        ixs
        for ixs in pyphi.utils.powerset(node_indices)
        if any([ix in purview for ix in ixs]) and not all([ix in purview for ix in ixs])
    ]
    return pyphi.models.CauseEffectStructure(
        [mice for mice in ces if mice.purview in purviews]
    )


def get_purview_context(ces, purview):
    return pyphi.models.CauseEffectStructure(
        set(
            [mice for mice in get_purview_subtext(ces, purview)]
            + [mice for mice in get_purview_supertext(ces, purview)]
            + [mice for mice in get_purview_paratext(ces, purview)]
        )
    )


def get_unit_subtext(ces, units):
    return pyphi.models.CauseEffectStructure(
        set(
            [mice for mice in get_mechanism_subtext(ces, units)]
            + [mice for mice in get_purview_subtext(ces, units)]
        )
    )


def get_unit_supertext(ces, units):
    return pyphi.models.CauseEffectStructure(
        set(
            [mice for mice in get_mechanism_supertext(ces, units)]
            + [mice for mice in get_purview_supertext(ces, units)]
        )
    )


def get_unit_paratext(ces, units):
    return pyphi.models.CauseEffectStructure(
        set(
            [mice for mice in get_mechanism_paratext(ces, units)]
            + [mice for mice in get_purview_paratext(ces, units)]
        )
    )


def get_unit_context(ces, units):
    return pyphi.models.CauseEffectStructure(
        set(
            [mice for mice in get_mechanism_context(ces, units)]
            + [mice for mice in get_purview_context(ces, units)]
        )
    )


def get_linked_ces(ces, subsystem):
    mechanisms = set([mice.mechanism for mice in ces])
    causes = [
        mice
        for mechanism in mechanisms
        for mice in ces
        if mice.mechanism == mechanism and mice.direction == CAUSE
    ]
    effects = [
        mice
        for mechanism in mechanisms
        for mice in ces
        if mice.mechanism == mechanism and mice.direction == EFFECT
    ]

    if not len(causes) == len(effects):
        print("Mismatching causes and effects! returning separated CES")
        return ces

    return pyphi.models.subsystem.CauseEffectStructure(
        [
            pyphi.models.mechanism.Concept(cause.mechanism, cause, effect, subsystem)
            for cause, effect in zip(causes, effects)
        ]
    )


def get_Phi_R(subset_distinctions, subset_relations):

    if hasattr(subset_distinctions[0], "direction"):
        print(
            "subset_distinctions CES must be concept style. Try passing to compute.get_linked_ces(ces, system) first!"
        )
        return (0, ((), ()))

    partitions = [
        part
        for part in pyphi.partition.bipartition(range(len(subset_distinctions)))
        if all([len(p) > 0 for p in part])
    ]

    Phi_R = np.inf
    MIP = ((), ())
    for partition in partitions:
        structure_0 = pyphi.models.subsystem.CauseEffectStructure(
            [subset_distinctions[i] for i in partition[0]]
        )
        structure_1 = pyphi.models.subsystem.CauseEffectStructure(
            [subset_distinctions[i] for i in partition[1]]
        )

        relations_0 = filter_relations(subset_relations, sep(structure_0))
        relations_1 = filter_relations(subset_relations, sep(structure_1))

        between_relations = [
            r for r in subset_relations if r not in relations_0 + relations_1
        ]

        lost_phi = sum([r.phi for r in between_relations])

        if lost_phi < Phi_R:
            Phi_R = lost_phi
            MIP = (
                [concept.mechanism for concept in structure_0],
                [concept.mechanism for concept in structure_1],
            )

    return (Phi_R, MIP)


def get_all_Phi_R(linked_ces, relations, system):
    if hasattr(linked_ces[0], "direction"):
        print(
            "subset_distinctions CES must be concept style. Try passing to compute.get_linked_ces(ces, system) first!"
        )
        return

    substructures = []
    for subset in tqdm(list(pyphi.utils.powerset(linked_ces, system, min_size=2))):
        linked = CES(subset)
        separated = sep(linked)
        subset_relations = compute.filter_relations(relations, separated)
        Phi_R, MIP = get_Phi_R(linked, subset_relations)

        substructures.append([Phi_R, MIP, separated])

    return substructures
