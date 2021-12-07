import pyphi
import itertools

from pyphi import relations as rels
from itertools import product
import numpy as np
from tqdm.auto import tqdm
import random
from operator import attrgetter

def add_node_labels(mice, system):
    mice.node_labels = tuple()
    return mice
    
def unfold_separated_ces(system):
    CAUSE = pyphi.direction.Direction(0)
    EFFECT = pyphi.direction.Direction(1)
    purviews = tuple(pyphi.utils.powerset(system.node_indices, nonempty=True))
    
    mices = [system.find_mice(direction, mechanism, purviews=purviews)
         for mechanism in pyphi.utils.powerset(system.node_indices, nonempty=True)
         for direction in [CAUSE, EFFECT]
        ]
    
    # remove any mice with phi=0
    mices = [mice for mice in mices if mice.phi>0]
    
    # kill unlinked mices
    causeeffect_mechanisms = set([mice.mechanism for mice in mices if mice.direction==CAUSE]).intersection(set([mice.mechanism for mice in mices if mice.direction==EFFECT]))
    filtered_ces = [mice for mice in mices if mice.mechanism in causeeffect_mechanisms]
    
    ces = pyphi.models.CauseEffectStructure(mices)
    for m in ces:
        m.node_labels = system.node_labels
    return ces
        
    
def compute_relations(subsystem, ces, max_k=3, num_relations=False):
    relations = []
    for k in range(2, max_k + 1):
        relata = [
            rels.Relata(subsystem, mices)
            for mices in itertools.combinations(ces, k)
            if all([mice.phi > 0 for mice in mices])
        ]

        if num_relations:
            relata = random.sample(relata, num_relations) 

        k_relations = [
            rels.relation(relatum)
            for relatum in (tqdm(relata) if len(relata) > 5000 else relata)
        ]
        k_relations = list(filter(lambda r: r.phi > 0, k_relations))
        relations.extend(k_relations)
    return relations




def compositional_state_from_system_state(state, system_indices=None):

    if system_indices==None:
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
            for subset_elements in pyphi.utils.powerset(
                system_indices, nonempty=True
            )
        }
        for state, direction in zip(
            [cause_state, effect_state],
            [pyphi.direction.Direction.CAUSE, pyphi.direction.Direction.EFFECT],
        )
    }


def add_missing_purviews(ces, filtered_ces):
    
    all_purviews = set([(mice.purview, mice.direction) for mice in ces])
    filtered_purviews = set([(mice.purview, mice.direction) for mice in filtered_ces])
    
    filtered_ces = list(filtered_ces)
    
    for purview, direction in all_purviews.difference(filtered_purviews):
        phi = 0
        missing_mice = None
        for mice in ces:
            if mice.purview == purview and mice.direction == direction and mice.phi>phi:
                missing_mice = mice
                
        filtered_ces.append(missing_mice)        
        
    return pyphi.models.CauseEffectStructure(filtered_ces)

def filter_ces_by_compositional_state(ces, compositional_state):

    # first separate the ces into mices and define the directions
    c = pyphi.direction.Direction.CAUSE
    e = pyphi.direction.Direction.EFFECT

    # next we run through all the mices and append any mice that has a state corresponding to the compositional state
    mices_with_correct_state = [
        mice
        for mice in ces
        if (
            tuple(mice.maximal_state[0])
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
    causeeffect_mechanisms = set([cause.mechanism for cause in causes]).intersection(set([effect.mechanism for effect in effects]))
    causeeffects = causes+effects
    filtered_ces = [mice for mice in causeeffects if mice.mechanism in causeeffect_mechanisms]
    
    return pyphi.models.CauseEffectStructure(filtered_ces)


def filter_relations(relations, filtered_ces):
    return list(
        filter(
            lambda r: all([relatum in filtered_ces for relatum in r.relata]), relations
        )
    )


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
        max_state = tuple(mice.maximal_state[0])
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
        for c_state, e_state in product(
            cause_states,
            effect_states,
        )
    ]
    return all_compositional_states


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
        for i, comp in (enumerate(tqdm(all_compositional_states) if len(all_compositional_states) > 100 else enumerate(all_compositional_states)))
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
        filtered_ces = maximal['ces']
        state = maximal['compositional_state']
        

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
        filtered_ces = maximal['ces']
        filtered_rels = maximal['relations']
        state = maximal['compositional_state']
        
    return (
        filtered_ces,
        filtered_rels,
        state,
    )


CAUSE = pyphi.direction.Direction(0)
EFFECT = pyphi.direction.Direction(1)


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


def distinction_touched(mice, part1, part2, direction):
    
    mechanism_in_part1 = any([m in part1 for m in mice.mechanism])
    purview_in_part2 = any([p in part2 for p in mice.purview])
    correct_direction = direction == mice.direction
    
    return mechanism_in_part1 and purview_in_part2 and correct_direction


def relation_untouched(untouched_ces, relation):
    relata_in_ces = all([relatum in untouched_ces for relatum in relation.relata])
    return relata_in_ces

def context_relations(relations, distinctions):
    return [relation for relation in relations 
            if (
                any([relatum in distinctions for relatum in relation.relata]) 
                and not all([relatum in distinctions for relatum in relation.relata])
            )
           ]
def get_big_phi(ces, relations, indices, partitions=None):
    
    phis = [
        min([cause.phi, effect.phi])
        for cause, effect in zip(
                [mice for mice in ces if mice.direction == CAUSE],
                [mice for mice in ces if mice.direction == EFFECT],
            )
    ]
    sum_of_small_phi = sum(phis) + sum([r.phi for r in relations])
    
    if len(indices)==1:
        return sum_of_small_phi, (((),()),'disintegration')

    if partitions == None:
        partitions = [
            part
            for part in pyphi.partition.bipartition(indices)
            if all([len(p) > 0 for p in part])
        ]
        
    min_phi = np.inf
    for parts in (tqdm(partitions, desc="System partitions") if len(partitions)>100 and len(indices)>4 else partitions):
        for p1, p2, direction in product(parts, parts, [CAUSE, EFFECT]):
            if not p1==p2:
                untouched_mices = pyphi.models.CauseEffectStructure(
                    [
                        mice
                        for mice in ces
                        if not distinction_touched(mice, p1, p2, direction)
                    ]
                )
                
                causeeffect_mechanisms = set(
                        [mice.mechanism for mice in untouched_mices if mice.direction==CAUSE]
                    ).intersection(
                        set([mice.mechanism for mice in untouched_mices if mice.direction==EFFECT])
                )
                untouched_ces = pyphi.models.CauseEffectStructure(
                    [
                        mice for mice in untouched_mices if mice.mechanism in causeeffect_mechanisms
                    ]
                )
                
                untouched_relations = [
                    r for r in relations if relation_untouched(untouched_ces, r)
                ]
                
                phis = [
                    min([cause.phi, effect.phi])
                    for cause, effect in zip(
                            [mice for mice in untouched_ces if mice.direction == CAUSE],
                            [mice for mice in untouched_ces if mice.direction == EFFECT],
                        )
                ]

                sum_phi_untouched = sum(phis) + sum(
                    [r.phi for r in untouched_relations]
                )

                lost_phi = sum_of_small_phi - sum_phi_untouched
                
                touched = [r for r in relations if r not in untouched_relations]
                if lost_phi < min_phi:
                    min_phi = lost_phi
                    min_cut = parts, p1, p2, direction

    big_phi = (sum_of_small_phi / 2 ** len(indices)) * (min_phi)
    return big_phi, min_cut


def compute_rels_and_ces_for_compositional_state(system, state, ces, max_k=3):

    if type(state) is tuple:
        state = compositional_state_from_system_state(state)
    filtered_ces = filter_ces_by_compositional_state(ces, state)
    relations = compute_relations(system, filtered_ces, max_k=max_k)
    
    return (
        filtered_ces,
        relations,
        state,
    )


def get_all_ces(system, ces=None):

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
        ) = compute_rels_and_ces_for_compositional_state(system, compositional_state, ces)
        phi, cut = get_big_phi(filtered_ces, filtered_relations, system.node_indices)
        
        all_ces.append({
                "ces": filtered_ces,
                "relations": filtered_relations,
                "compositional_state": compositional_state,
                "big phi": phi,
        })

    return all_ces


def get_maximal_ces(system, ces=None, max_k=3):

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

    compositional_states = get_all_compositional_states(ces)
    
    big_phi = 0
    for compositional_state in compositional_states if len(compositional_states)<2 else tqdm(compositional_states, desc='Computing Big Phi for all compositional states'):
        (
            filtered_ces,
            filtered_relations,
            compositional_state,
        ) = compute_rels_and_ces_for_compositional_state(system, compositional_state, ces, max_k)
        phi, cut = get_big_phi(filtered_ces, filtered_relations, system.node_indices)

        if phi >= big_phi:
            maximal = {
                "ces": filtered_ces,
                "relations": filtered_relations,
                "compositional_state": compositional_state,
                "big phi": phi,
                "MIP": cut,
            }
            big_phi = phi

    return maximal

def get_untouced_ces_and_rels(ces,relations,parts):
    untouched_ces = pyphi.models.CauseEffectStructure([
                    mice
                    for mice in ces
                    if not distinction_touched(
                        mice, parts[1], parts[2], parts[3]
                    )
                ])
    untouched_relations = [r for r in relations if relation_untouched(untouched_ces, r)]
            
    return untouched_ces, untouched_relations



def get_component_phi(ces, relations, component_distinctions):

    component_ces = [mice for mice, d in zip(ces, component_distinctions) if d]
    component_relations = [
        r for r in relations if relation_untouched(component_ces, r)
    ]

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
            r
            for r in component_relations
            if relation_untouched(untouched_ces, r)
        ]

        sum_phi_untouched = sum([mice.phi for mice in untouched_ces]) + sum(
            [r.phi for r in untouched_relations]
        ) if len(untouched_relations)>0 else sum([mice.phi for mice in untouched_ces])
        
        max_possible_phi_untouched = sum(
            [len(mice.mechanism) for mice in untouched_ces]
        ) + sum([min([len(relatum.purview) for relatum in relation.relata])
                    for relation in untouched_relations]
            )

        lost_phi = sum_of_small_phi - sum_phi_untouched

        if lost_phi < min_phi:
            min_phi = lost_phi
            dominant_distinction = component_ces[i]
            component_phi = sum_phi_untouched / max_possible_phi_untouched * lost_phi

    return component_phi, dominant_distinction
