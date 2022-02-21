import pyphi
from pyphi.models.subsystem import FlatCauseEffectStructure as sep
from pyphi.models.subsystem import CauseEffectStructure as CES
from pyphi.big_phi import (
    all_nonconflicting_distinction_sets as d_sets,
    informativeness,
    selectivity,
)

from visualize_pyphi import compute

import toolz
import random
import numpy as np

from tqdm.auto import tqdm
import itertools
from visualize_pyphi import compute

directions = [pyphi.Direction.CAUSE, pyphi.Direction.EFFECT]
C, E = directions
from itertools import product


directions = [pyphi.Direction.CAUSE, pyphi.Direction.EFFECT]
CAUSE = pyphi.Direction.CAUSE
EFFECT = pyphi.Direction.EFFECT


def get_all_miws(subsystem):
    pset = list(pyphi.utils.powerset(subsystem.node_indices, nonempty=True))
    mechanisms = pset
    purviews = pset

    all_miws = []

    for mechanism in mechanisms:
        for direction in directions:
            candidate_mices = [
                subsystem.find_mice(direction, mechanism, (purview,))
                for purview in purviews
            ]
            # check if each mice is maximally irreducible within
            candidate_mices_MIW = [
                mice
                for mice in candidate_mices
                if mice.phi > 0
                and not any(
                    [
                        all([unit in mice.purview for unit in mice2.purview])
                        and mice2.phi > mice.phi
                        for mice2 in candidate_mices
                    ]
                )
            ]

            all_miws.extend(candidate_mices_MIW)
    return all_miws


def separate_miws_for_direction(miws):
    return [[m for m in miws if m.direction == d] for d in directions]


def get_empty_purv_slots(subsystem, max_mices_with_winners=None):
    slots = [list(pyphi.utils.powerset(subsystem.node_indices, nonempty=True))] * 2
    if max_mices_with_winners:
        slots = [
            [slot for slot in slots[d] if slot not in [m.purview for m in direction]]
            for d, direction in enumerate(max_mices_with_winners)
        ]
    return slots


def get_mices_over_purview_slots(slots, miws, subsystem):
    return [
        [
            [
                mice
                for mice in miws[d]
                if mice.purview == slot and mice.direction == direction
            ]
            for slot in slots[d]
        ]
        for d, direction in enumerate(directions)
    ]


def get_max_mices_over_purview_slots(mices_over_purview_slots):
    return [
        [max(mices_over_slot) for mices_over_slot in direction if mices_over_slot]
        for direction in mices_over_purview_slots
    ]


def get_conflicts(max_mices_over_purview_slots, subsystem):
    slots = list(pyphi.utils.powerset(subsystem.node_indices, nonempty=True))
    mices_with_same_mechanism = [
        [
            [
                mice
                for mice in max_mices_over_purview_slots[d]
                if mice.mechanism == slot and mice.direction == direction
            ]
            for slot in slots
        ]
        for d, direction in enumerate(directions)
    ]

    conflicts = [
        [mices for mices in direction if len(mices) > 1]
        for direction in mices_with_same_mechanism
    ]
    return conflicts


def get_winners(conflicts):
    winners = []
    for direction in conflicts:
        direction_winners = []
        for conflict in direction:
            purv_length = [len(mice.purview) for mice in conflict]
            if len(set(purv_length)) > 1:
                max_length = max(purv_length)
                winner = [mice for mice in conflict if len(mice.purview) == max_length]
                direction_winners.extend(winner)
            else:
                winner = [mice for mice in conflict if mice.phi == max(conflict).phi]
                #                 if len(winner)>1:
                #                     winner = winner[0]
                direction_winners.extend(winner)
        winners.append(direction_winners)
    return winners


def merge_directional_mice_list(mice_list):
    return sep(list(toolz.concat(mice_list)))


def prune_unlinked_mices(mices):
    mices = merge_directional_mice_list(mices)
    return pyphi.models.FlatCauseEffectStructure(
        [
            mice
            for mice in mices
            if len([m for m in mices if m.mechanism == mice.mechanism]) == 2
            and len(set([m.direction for m in mices if m.mechanism == mice.mechanism]))
            == 2
        ]
    )


def get_remaining_miws(miws, max_mices_with_winners=None):
    if max_mices_with_winners:
        used_mechanisms = [
            [m.mechanism for m in direction] for direction in max_mices_with_winners
        ]

        used_purviews = [
            [m.purview for m in direction] for direction in max_mices_with_winners
        ]

        remaining_miws = []
        for d, direction in enumerate(directions):
            remaining_mices_for_direction = []
            for mice in miws[d]:
                if (
                    mice.mechanism not in used_mechanisms[d]
                    and mice.purview not in used_purviews[d]
                ):
                    remaining_mices_for_direction.append(mice)
            remaining_miws.append(remaining_mices_for_direction)

        return remaining_miws
    else:
        return miws


def fill_slots_bestbiggest(subsystem):

    slots = get_empty_purv_slots(subsystem)
    miws = separate_miws_for_direction(get_all_miws(subsystem))
    max_mices_with_winners = [[], []]

    while miws[0] and miws[1]:
        slots = get_empty_purv_slots(subsystem, max_mices_with_winners)
        miws = get_remaining_miws(miws, max_mices_with_winners)
        mices_over_purview_slots = get_mices_over_purview_slots(slots, miws, subsystem)
        max_mices_over_purview_slots = get_max_mices_over_purview_slots(
            mices_over_purview_slots
        )
        max_mices_over_purview_slots = [
            max_mices_over_purview_slots[d] + max_mices_with_winners[d]
            for d in range(len(directions))
        ]

        conflicts = get_conflicts(max_mices_over_purview_slots, subsystem)

        winners = get_winners(conflicts)

        max_mices_with_winners = [
            [
                m
                for m in max_mices_over_purview_slots[i]
                if m not in list(toolz.concat(conflicts[i]))
            ]
            + winners[i]
            for i in range(len(winners))
        ]
    else:
        final_ces = prune_unlinked_mices(max_mices_with_winners)
        final_ces.subsystem = subsystem
        return final_ces


# ---------------------
# Reflexivity first


def make_bag(all_mices):
    bag = dict()
    for mice in all_mices:
        m = mice.mechanism
        d = mice.direction
        p = mice.purview

        if m in bag.keys():
            if d in bag[m].keys():
                bag[m][d][p] = mice
            else:
                bag[m][d] = dict()
                bag[m][d][p] = mice
        else:
            bag[m] = dict()
            bag[m][d] = dict()
            bag[m][d][p] = mice

    return bag


def get_bag_of_mices(subsystem, mechanisms, purviews, candidate="irreducible"):

    if candidate == "irreducible":
        return {
            mechanism: {
                direction: {
                    purview: subsystem.find_mice(direction, mechanism, (purview,))
                    for purview in purviews
                }
                for direction in directions
            }
            for mechanism in mechanisms
        }

    elif candidate == "miw":

        # compute candidate mices for every purview
        all_mices = []

        for mechanism in mechanisms:
            for direction in directions:
                candidate_mices = [
                    subsystem.find_mice(direction, mechanism, (purview,))
                    for purview in purviews
                ]
                # check if each mice is maximally irreducible within
                candidate_mices_MIW = [
                    mice
                    for mice in candidate_mices
                    if mice.phi > 0
                    and not any(
                        [
                            all([unit in mice.purview for unit in mice2.purview])
                            and mice2.phi > mice.phi
                            for mice2 in candidate_mices
                        ]
                    )
                ]

                all_mices.extend(candidate_mices_MIW)

        bag = make_bag(all_mices)
        return bag


def get_mices_from_bag(bag_of_mices, subsystem):
    pset = list(pyphi.utils.powerset(subsystem.node_indices, nonempty=True))
    bag = []
    for m in pset:
        for d in directions:
            for p in pset:
                try:
                    bag.append(bag_of_mices[m][d][p])
                except:
                    pass
    return bag


def fill_slots_reflexive(subsystem):

    pset = list(pyphi.utils.powerset(subsystem.node_indices, nonempty=True))
    bag_of_mices = get_bag_of_mices(subsystem, pset, pset, candidate="miw")
    flat_bag = []
    for mechanism in pset:
        for direction in directions:
            for purview in pset:
                try:
                    flat_bag.append(bag_of_mices[mechanism][direction][purview])
                except:
                    pass

    reflexive_ds = [
        pyphi.subsystem.Concept(m1.mechanism, m1, m2, subsystem)
        for m1, m2 in itertools.combinations(flat_bag, 2)
        if m1.mechanism == m2.mechanism == m1.purview == m2.purview
        and m1.direction == C
        and m2.direction == E
    ]

    reflexive_sep = sep(pyphi.models.CauseEffectStructure(reflexive_ds))

    reflexive_causes = [m.purview for m in reflexive_sep if m.direction == C]
    reflexive_effects = [m.purview for m in reflexive_sep if m.direction == E]
    reflexive_mechanisms = [m.mechanism for m in reflexive_sep]

    remaining_mices = []
    for m in flat_bag:
        if (
            m.direction == C
            and m.mechanism not in reflexive_mechanisms
            and m.purview not in reflexive_causes
        ):
            remaining_mices.append(m)
        if (
            m.direction == E
            and m.mechanism not in reflexive_mechanisms
            and m.purview not in reflexive_effects
        ):
            remaining_mices.append(m)

    remaining_ces = compute.CES_from_bag_of_mices(make_bag(remaining_mices))

    final_ces = compute.get_linked_ces(
        list(reflexive_sep) + list(remaining_ces), subsystem
    )
    final_ces.subsystem = subsystem
    return final_ces


def count_purview_element_states(mices, subsystem):
    # Returns a list containing the number of purview elements in each state
    # Example output: [[# node1 OFF, # node1 ON], [# node2 OFF, # node2 ON], ...]
    return [
        [
            len(
                [
                    mice
                    for mice in mices
                    if node in mice.purview
                    and mice.specified_state[0][mice.purview.index(node)] == state
                ]
            )
            for state in [0, 1]
        ]
        for node in subsystem.node_indices
    ]


### ----- Random search algorithm


def get_candidate_distinctions(bag_of_mices, subsystem):

    return [
        [
            pyphi.models.mechanism.Concept(
                mech,
                bag_of_mices[mech][CAUSE][cause],
                bag_of_mices[mech][EFFECT][effect],
                subsystem,
            )
            for cause, effect in product(
                *[bag_of_mices[mech][CAUSE], bag_of_mices[mech][EFFECT]]
            )
        ]
        for mech in bag_of_mices.keys()
    ]


def large_purview_candidate_distinction_set(candidate_distinctions):
    distinctions = []
    for candidates in candidate_distinctions:
        summed_purview_length = [
            sum([len(d.cause.purview), len(d.effect.purview)]) for d in candidates
        ]
        # pick a distinction that has maximally large purviews
        distinctions.append(
            candidates[
                random.choice(
                    [
                        i
                        for i, x in enumerate(summed_purview_length)
                        if x == max(summed_purview_length)
                    ]
                )
            ]
        )
    return distinctions


def swap_distinctions(subsystem, distinctions, candidates, n):

    # swap out n distinctions randomly
    to_swap = random.sample(range(len(distinctions)), n)
    new_distinctions = [
        distinction if not i in to_swap else random.choice(candidates[i])
        for i, distinction in enumerate(distinctions)
    ]

    return CES(new_distinctions, subsystem)


def minimize_conflicts(
    subsystem,
    candidate_distinctions,
    bag_of_mices,
    distinctions_to_swap=3,
    max_conflicts_accepted=100,
    convergence=100,
    max_attempts=10,
):

    n_conflicts = np.inf
    attempts = 0
    while n_conflicts > max_conflicts_accepted and attempts < max_attempts:
        # intialize CES with distinctions with as large purviews as possible
        candidate_ces = CES(
            large_purview_candidate_distinction_set(candidate_distinctions)
        )

        # intial conflicts
        n_conflicts = len(list(d_sets(candidate_ces)))

        # iterate, swap out distinctions, accept swap if conflicts fall
        stuck = 0
        while stuck < convergence:
            new_ces = swap_distinctions(
                subsystem,
                candidate_ces,
                candidate_distinctions,
                random.choice(range(distinctions_to_swap)),
            )
            new_conflicts = len(list(d_sets(new_ces)))

            if new_conflicts < n_conflicts:
                candidate_ces = new_ces
                n_conflicts = new_conflicts
                stuck = 0

            else:
                stuck += 1

    # swapping purviews and mechanisms and picking the one that maximizes um of sum of small_phi for distinctions
    # swapped = swap_purviews(subsystem,candidate_ces,bag_of_mices)
    # mechanism_swapped = list(set([swap_mechanisms(subsystem, ces, bag_of_mices) for ces in swapped]))
    # candidate_ces = max(mechanism_swapped,key=lambda ces: estimate_congruence(ces, subsystem))
    candidate_ces.subsystem = subsystem
    return candidate_ces


def get_candidate_CESs(
    subsystem,
    candidate_distinctions,
    bag_of_mices,
    distinctions_to_swap=5,
    max_conflicts_accepted=100,
    convergence=100,
    runs=100,
):
    candidates = [
        minimize_conflicts(
            subsystem,
            candidate_distinctions,
            bag_of_mices,
            distinctions_to_swap=distinctions_to_swap,
            max_conflicts_accepted=max_conflicts_accepted,
            convergence=convergence,
        )
        for i in tqdm(
            range(runs), desc="finding minimally conflicting candidate distinction sets"
        )
    ]
    print("swapping purviews and mechanisms")
    p_swapped = [
        swap_purviews(
            subsystem,
            candidate,
            bag_of_mices,
        )
        for candidate in candidates
    ]
    m_swapped = [
        swap_mechanisms(
            subsystem,
            ces,
            bag_of_mices,
        )
        for cess in p_swapped
        for ces in cess
    ]

    cess = []
    for ces in m_swapped:
        ces.subsystem = subsystem
        cess.append(ces)
    return cess


def get_candidate_structures(candidate_cess, subsystem, max_degree=3, max_cess=10):

    structures = list(
        set(
            [
                distinctions
                for all_distinctions in candidate_cess
                for distinctions in list(d_sets(all_distinctions))
            ]
        )
    )

    max_len = max([len(ces) for ces in structures])
    max_len_cess = list(filter(lambda ces: len(ces) == max_len, structures))

    subsystem.clear_caches()
    return [
        pyphi.big_phi.PhiStructure(
            ces,
            pyphi.relations.relations(
                subsystem, ces, max_degree=max_degree, progress=False
            ),
        )
        for ces in tqdm(
            random.sample(max_len_cess, 10), desc="Computing Phi structures"
        )
    ]


def get_specific_structure(PhiStructures):

    info = [
        phi_structure.system_intrinsic_information()
        for phi_structure in tqdm(PhiStructures)
    ]

    return PhiStructures[info.index(max(info))]


def get_purview_specifiers(bag_of_mices):
    specifiers = dict()
    for mech, directions in bag_of_mices.items():
        for direction, mices in directions.items():
            for purview, mice in mices.items():
                if (purview, direction) in specifiers.keys():
                    specifiers[(purview, direction)].append(mech)
                else:
                    specifiers[(purview, direction)] = [mech]
    return specifiers


def swap_purviews(subsystem, distinctions, bag_of_mices, shuffles=100):

    # flatten the ces to run through every mice independently
    flat_ces = [mice for mice in sep(distinctions)]
    mechanisms = set([m.mechanism for m in flat_ces])

    # get all the mechanisms that specify every purvie-direction
    specifiers = get_purview_specifiers(bag_of_mices)

    # run the swapping algorithme for "shuffles" number of times, to avoid ordering issues
    candidate_sets = []
    for i in range(shuffles):

        # shuffle the list of mices, to swap in different orders each time
        random.shuffle(flat_ces)

        # define lists that will keep track of distinctions, mechanisms-directions and purview-directions that are specified
        distinctions = []
        taken_mechanism = []
        taken_purview = []

        # loop through every mice in the ces
        for mice in flat_ces:

            # chcek that its mechanism-direction and purview-direction is not already picked/swapped
            if (
                not (mice.mechanism, mice.direction) in taken_mechanism
                and not (mice.purview, mice.direction) in taken_purview
            ):
                # finding which other mechanisms can specify the purview-direction specified by the current mice
                candidate_specifiers = []
                for potential_candidate in specifiers[(mice.purview, mice.direction)]:

                    # picking out the mice that would potentially swap in
                    candidate_mice = bag_of_mices[potential_candidate][mice.direction][
                        mice.purview
                    ]

                    # given another candidate specifier, check that it is valid for swapping:
                    # the candidate mechanism is actually a mechanism in the orignal CES
                    # the candidate mechanism is not the same as the original mechanism
                    # the mice mechanism specifies the purvie the candidate originally specified
                    # the candidate mechanism-direction must not already have been taken
                    # the candidate phi must be higher than the original mice for both swaps
                    # the purview-direction specified by the candidate originally must not already be taken
                    if (
                        candidate_mice.mechanism in mechanisms
                        and not mice.mechanism == candidate_mice.mechanism
                        and not (candidate_mice.mechanism, candidate_mice.direction)
                        in taken_mechanism
                        # and candidate_mice.phi > mice.phi
                    ):

                        # finding the second mice that would be swapped
                        orig_mice = [
                            m
                            for m in flat_ces
                            if m.mechanism == candidate_mice.mechanism
                            and m.direction == candidate_mice.direction
                        ][0]

                        # check if the mice.mechanism specifies the caandidate mechanism's orignal purview
                        if (
                            mice.mechanism
                            in specifiers[(orig_mice.purview, orig_mice.direction)]
                        ):
                            new_mice = bag_of_mices[mice.mechanism][
                                candidate_mice.direction
                            ][orig_mice.purview]
                            if (
                                not (
                                    new_mice.purview,
                                    new_mice.direction,
                                )
                                in taken_purview
                                # and new_mice.phi > orig_mice.phi
                            ):
                                candidate_specifiers.append(candidate_mice)

                if len(candidate_specifiers) > 0:
                    swapper = random.choice(candidate_specifiers)
                    swapped = [
                        m
                        for m in flat_ces
                        if m.mechanism == swapper.mechanism
                        and m.direction == swapper.direction
                    ][0]
                    taken_mechanism.append((swapper.mechanism, swapper.direction))
                    taken_purview.append((swapper.purview, swapper.direction))
                    distinctions.append(swapper)

                    new_mice = bag_of_mices[mice.mechanism][mice.direction][
                        swapped.purview
                    ]
                    taken_mechanism.append((new_mice.mechanism, new_mice.direction))
                    taken_purview.append((new_mice.purview, new_mice.direction))
                    distinctions.append(new_mice)

                else:
                    taken_mechanism.append((mice.mechanism, mice.direction))
                    taken_purview.append((mice.purview, mice.direction))
                    distinctions.append(mice)

        final_ces = compute.get_linked_ces(distinctions, subsystem)
        final_ces.subsystem = subsystem
        candidate_sets.append(final_ces)

    return candidate_sets


def swap_mechanisms(subsystem, distinctions, bag_of_mices):

    missing_mechanisms = list(
        set(bag_of_mices.keys()) - set([d.mechanism for d in distinctions])
    )

    new_distinctions = []
    for distinction in distinctions:
        cause_purview = distinction.cause.purview
        effect_purview = distinction.effect.purview

        existing_new_mechanisms = [d.mechanism for d in new_distinctions]

        potential_new_distinctions = []
        for mech, directions in bag_of_mices.items():
            if (
                mech in missing_mechanisms
                and mech not in existing_new_mechanisms
                and cause_purview in directions[distinction.cause.direction].keys()
                and effect_purview in directions[distinction.effect.direction].keys()
                and min(
                    [
                        directions[distinction.cause.direction][cause_purview].phi,
                        directions[distinction.effect.direction][effect_purview].phi,
                    ]
                )
                > distinction.phi
            ):
                potential_new_distinctions.append(
                    pyphi.models.mechanism.Concept(
                        mech,
                        directions[distinction.cause.direction][cause_purview],
                        directions[distinction.effect.direction][effect_purview],
                        subsystem,
                    )
                )
        if len(potential_new_distinctions) > 0:
            swapped = random.choice(potential_new_distinctions)
            new_distinctions.append(swapped)
            missing_mechanisms.pop(missing_mechanisms.index(swapped.mechanism))
            missing_mechanisms.append(distinction.mechanism)
        else:
            new_distinctions.append(distinction)

    return CES(new_distinctions)


def random_search_for_sia(
    subsystem,
    mechanisms=None,
    purviews=None,
    bag_of_mices=None,
    candidate_purviews="miw",
    max_conflicts_accepted=15,
    distinctions_to_swap=10,
    convergence=100,
    runs=100,
    max_degree=3,
):
    mechanisms = (
        list(pyphi.utils.powerset(subsystem.node_indices, nonempty=True))
        if mechanisms == None
        else mechanisms
    )
    purviews = (
        list(pyphi.utils.powerset(subsystem.node_indices, nonempty=True))
        if purviews == None
        else purviews
    )
    bag_of_mices = (
        get_bag_of_mices(subsystem, mechanisms, purviews, candidate=candidate_purviews)
        if bag_of_mices == None
        else bag_of_mices
    )

    candidate_distinctions = get_candidate_distinctions(bag_of_mices, subsystem)
    candidate_cess = get_candidate_CESs(
        subsystem,
        candidate_distinctions,
        bag_of_mices,
        distinctions_to_swap=distinctions_to_swap,
        max_conflicts_accepted=max_conflicts_accepted,
        convergence=convergence,
        runs=runs,
    )
    candidate_structures = get_candidate_structures(
        candidate_cess, subsystem, max_degree=max_degree
    )
    specific_structure = get_specific_structure(candidate_structures)
    return pyphi.big_phi.sia(
        subsystem, specific_structure.distinctions, specific_structure.relations
    )


def count_purview_element_states_in_ces(CES, subsystem):
    # Returns a list containing the number of purview elements in each state
    # Example output: [[# node1 OFF, # node1 ON], [# node2 OFF, # node2 ON], ...]
    return [
        [
            len(
                [
                    mice
                    for distinction in CES
                    for mice in [distinction.cause, distinction.effect]
                    if node in mice.purview
                    and mice.specified_state[0][mice.purview.index(node)] == state
                ]
            )
            for state in [0, 1]
        ]
        for node in subsystem.node_indices
    ]


def estimate_congruence(CES, subsystem):
    return sum(
        [
            state
            for purview in count_purview_element_states_in_ces(CES, subsystem)
            for state in purview
        ]
    )


def get_bigphi_supertext_approximation(subsystem, purview_element_count):
    # For now this assumes all purvs are in the same state, only pass purview_element_count as a list of ints (one per node)
    n = subsystem.size
    num = sum(2 ** purview_element_count[n] for n in range(n))
    #     print(num)
    return (num / (n * 2 ** 2 ** n)) * 2 ** min(purview_element_count)


def get_existence_approximation(subsystem, ces):
    # For now this assumes all purvs are in the same state, only pass purview_element_count as a list of ints (one per node)
    purview_element_count = count_purview_element_states_in_ces(ces, subsystem)
    purview_element_count = [p[0] for p in purview_element_count]
    n = subsystem.size
    num = sum(2 ** purview_element_count[n] for n in range(n))
    #     print(num)
    return num / (n * 2 ** 2 ** n)


def get_num_potential_relations(n):
    return n * (2 ** 2 ** n)


def sample_relations(subsystem, ces, sample_size=10, print_progress=False):
    degree = (
        max([p[0] for p in count_purview_element_states_in_ces(ces, subsystem)]) // 2
    )
    if print_progress:
        print(f"Listing non-empty {degree}-degree overlaps...")
    potential_relata = list(
        pyphi.relations.potential_relata(
            subsystem, sep(ces), min_degree=degree, max_degree=degree
        )
    )
    keep_sampling = True
    sample_n = 0
    if print_progress:
        print("Sampling...")

    while keep_sampling:
        sample_n += 1
        if print_progress:
            print(sample_n)
        sample_relata = list(
            toolz.concat(
                [random.sample(potential_relata, 1) for n in range(sample_size)]
            )
        )

        relations = [pyphi.relations.relation(r) for r in sample_relata]
        rphis = [r.phi for r in relations]

        if min(rphis) > 0:
            keep_sampling = False

        avg_phi = sum(r.phi for r in relations) / len(relations)
    return avg_phi


def get_keikoreza_bigphi(
    subsystem,
    ces,
    relation_sample_size=500,
    compute_real_rphi_avg=False,
    avg_phi=None,
    degrees=None,
):
    # For now this assumes all purvs are in the same state, only pass purview_element_count as a list of ints (one per node)
    n = subsystem.size
    purview_element_count = count_purview_element_states_in_ces(ces, subsystem)
    purview_element_count = [p[0] for p in purview_element_count]
    min_purv_count = purview_element_count.index(min(purview_element_count))

    if not avg_phi:
        if compute_real_rphi_avg:
            relations = list(pyphi.relations.all_relations(subsystem, ces))
            possible_rels = len(subsystem) * 2 ** 2 ** len(subsystem)
            avg_phi = sum(r.phi for r in relations) / possible_rels
        else:
            # CHECK: should random sample start from degree 2 or try 1 rels?
            if not degrees:
                degrees = [
                    random.randint(2, len(sep(ces)))
                    for s in range(relation_sample_size)
                ]
            else:
                degrees = list(
                    toolz.concat(
                        [random.sample(degrees, 1) for s in range(relation_sample_size)]
                    )
                )
            samples = [random.sample(sep(ces), d) for d in degrees]
            sample_relata = [
                pyphi.relations.Relata(subsystem, sample) for sample in samples
            ]
            # print(sample_relata)

            relations = [pyphi.relations.relation(r) for r in sample_relata]
            avg_phi = sum(r.phi for r in relations) / len(relations)
    print(avg_phi)

    least_appearing_purview_element = tuple([subsystem.node_indices[min_purv_count]])
    partitions = list(
        pyphi.partition.system_temporal_directed_bipartitions_cut_one(
            subsystem.node_indices
        )
    )
    least_appearing_element_partitions = [
        p
        for p in partitions
        if least_appearing_purview_element == p.from_nodes
        or least_appearing_purview_element == p.to_nodes
    ]
    partitioned_cess = [
        [c for c in ces if c in pyphi.big_phi.unaffected_distinctions(ces, p)]
        for p in least_appearing_element_partitions
    ]
    untouched_purview_element_counts = [
        count_purview_element_states_in_ces(ces, subsystem) for ces in partitioned_cess
    ]

    untouched_purview_element_counts_sums = [
        sum(toolz.concat(count_purview_element_states_in_ces(ces, subsystem)))
        for ces in partitioned_cess
    ]

    untouched_counts_max = untouched_purview_element_counts[
        untouched_purview_element_counts_sums.index(
            max(untouched_purview_element_counts_sums)
        )
    ]
    untouched_counts_max = [c[0] for c in untouched_counts_max]

    selectivity = avg_phi
    informativeness = avg_phi * (
        sum(2 ** purview_element_count[n] for n in range(n))
        - sum(2 ** untouched_counts_max[n] for n in range(n))
    )

    return selectivity * informativeness


def compute_possible_number_of_relations(subsystem):
    return len(subsystem) * (2 ** 2 ** len(subsystem))


def compute_selectivity(subsystem, relations):
    phi_sum = sum([r.phi for r in relations])
    possible_rels_n = compute_possible_number_of_relations(subsystem)
    return phi_sum / possible_rels_n


def compute_existence(subsystem, ces, relations, selectivity=None):
    if not selectivity:
        if not relations:
            relations = pyphi.relations.relations(subsystem, ces)
        selectivity = compute_selectivity(subsystem, relations)
    phi_sum = sum([r.phi for r in relations])
    possible_rels_n = compute_possible_number_of_relations(subsystem)
    return selectivity * phi_sum


def estimate_existence(subsystem, ces):
    purview_counts = [p[0] for p in count_purview_element_states_in_ces(ces, subsystem)]
    degree = max(purview_counts)
    avg_phi = sample_relations(subsystem, ces)
    existing_rels_num = estimate_existing_relations_number(subsystem, ces)
    sum_phi = avg_phi * existing_rels_num
    all_rels_num = compute_possible_number_of_relations(subsystem)
    selectivity = sum_phi / all_rels_num
    return selectivity * sum_phi


def estimate_existing_relations_number(subsystem, ces):
    purview_counts = [p[0] for p in count_purview_element_states_in_ces(ces, subsystem)]
    return sum([2 ** n for n in purview_counts])


def resolve_conflicts_biggest_highest(subsystem, ces):

    pset = list(pyphi.utils.powerset(subsystem.node_indices, nonempty=True))

    non_unique_causes = [
        p for p in pset if len([d for d in ces if d.cause_purview == p]) > 1
    ]
    non_unique_effects = [
        p for p in pset if len([d for d in ces if d.effect_purview == p]) > 1
    ]

    conflicts = [
        d
        for d in ces
        if d.cause_purview in non_unique_causes
        or d.effect_purview in non_unique_effects
    ]

    while conflicts:

        non_unique_causes = [
            p for p in pset if len([d for d in ces if d.cause_purview == p]) > 1
        ]
        non_unique_effects = [
            p for p in pset if len([d for d in ces if d.effect_purview == p]) > 1
        ]

        conflicts = [
            d
            for d in ces
            if d.cause_purview in non_unique_causes
            or d.effect_purview in non_unique_effects
        ]

        non_conflicting_distinctions = [d for d in ces if d not in conflicts]

        survivors = []
        for d1 in conflicts:
            competitors = [
                d
                for d in conflicts
                if d.cause_purview == d1.cause_purview
                or d.effect_purview == d1.effect_purview
                and d1.mechanism != d.mechanism
            ]
            competitor_length = [
                (len(d.cause_purview) + len(d.effect_purview)) for d in competitors
            ]
            if len(set(competitor_length)) > 1:
                max_length = max(competitor_length)
                winner = [
                    d
                    for d in competitors
                    if len(d.cause_purview) + len(d.effect_purview) == max_length
                ]
                survivors.extend(winner)
            else:
                winner = [d for d in competitors if d.phi == max(competitors).phi]
                if len(winner) == 1:
                    survivors.extend(winner)
                else:
                    survivors.append(winner[0])

        survivors = list(set(survivors))
        ces = pyphi.models.CauseEffectStructure(
            non_conflicting_distinctions + survivors
        )
    return ces
