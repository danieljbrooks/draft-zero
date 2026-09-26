"""Bad-target effect classification. In the exp #2 pilot 79% of targeting prompts were
unclassifiable (docs/006); these are the cases that fixed most of that, and the traps that
would turn a legitimate own-card choice into a false "bad target"."""
import pytest

from draftzero.metrics import classify_effect, rules_restricted, targets_anything


@pytest.mark.parametrize("text, expected", [
    # one required target; "up to one target" and the word "targets" don't count
    ("Fiery Annihilation deals 5 damage to target creature. Exile up to one target Equipment "
     "attached to that creature.", "harmful"),
    ("This spell costs {3} less to cast if it targets a tapped creature.\nDestroy target creature.", "harmful"),
    # reminder text is not rules text
    ("Destroy target creature. Create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice "
     "this token: You gain 3 life.\")", "harmful"),
    # Auras target when cast, though the text never says so
    ("Enchant creature\nEnchanted creature gets +1/+1 for each Forest you control.", "beneficial"),
    ("Enchant creature\nEnchanted creature loses all abilities and is a green and white Citizen.", "harmful"),
    ("Target creature gets +3/+3 until end of turn.", "beneficial"),
    ("Counter target spell.", "harmful"),
])
def test_classified(text, expected):
    assert classify_effect(text) == expected


@pytest.mark.parametrize("text", [
    # a second choice logged under the same source: picking your own card there is fine
    "Counter target spell. Draw a card, then discard a card.",
    "Target creature's owner puts it on their choice of the top or bottom of their library.\nSurveil 1.",
    # two targets, one per side
    "Target creature you control deals damage equal to its power to target creature you don't control.",
    # the pick is a cost, not a target
    "As an additional cost to cast this spell, sacrifice a creature.\nExile target creature.",
])
def test_ambiguous_stays_unclassified(text):
    assert classify_effect(text) is None


def test_rules_restricted_and_non_targets():
    assert rules_restricted("When {this} enters, return target creature an opponent controls to its owner's hand.")
    assert not targets_anything("When {this} enters, draw two cards, then discard two cards.")
    assert targets_anything("Enchant creature\nEnchanted creature gets +2/+2.")
    assert targets_anything("Destroy target creature.")
