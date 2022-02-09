import pyphi
import toolz
from pyphi.models.subsystem import FlatCauseEffectStructure as sep

directions = [pyphi.Direction.CAUSE, pyphi.Direction.EFFECT]


def get_all_miws(subsystem):
    pset = list(pyphi.utils.powerset(subsystem.node_indices, nonempty=True))
    mechanisms = pset
    purviews = pset

    all_miws = []

    for mechanism in tqdm(mechanisms):
        for direction in [CAUSE, EFFECT]:
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
    directions = [pyphi.Direction.CAUSE, pyphi.Direction.EFFECT]
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
    directions = [pyphi.Direction.CAUSE, pyphi.Direction.EFFECT]
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


def fill_slots(subsystem):

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
                for direction in [
                    C,
                    E,
                ]
            }
            for mechanism in mechanisms
        }

    elif candidate == "miw":

        # compute candidate mices for every purview
        all_mices = []

        for mechanism in tqdm(mechanisms):
            for direction in [C, E]:
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
    directions = [pyphi.Direction.CAUSE, pyphi.Direction.EFFECT]
    bag = []
    for m in pset:
        for d in directions:
            for p in pset:
                try:
                    bag.append(bag2[m][d][p])
                except:
                    pass
    return bag


def fill_slots_reflexive(subsystem):

    pset = list(pyphi.utils.powerset(subsystem.node_indices, nonempty=True))
    bag_of_mices = get_bag_of_mices(subsystem, pset, pset, candidate="miw")
    flat_bag = []
    for mechanism in tqdm(pset):
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

    remaining_ces = CES_from_bag_of_mices(make_bag(remaining_mices))

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
