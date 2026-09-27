package org.draftzero.mzbridge;

import mage.abilities.Ability;
import mage.abilities.DelayedTriggeredAbility;
import mage.abilities.SpellAbility;
import mage.abilities.effects.ContinuousEffectImpl;
import mage.abilities.effects.common.SacrificeTargetEffect;
import mage.abilities.keyword.EnchantAbility;
import mage.constants.Duration;
import mage.constants.Layer;
import mage.constants.Outcome;
import mage.constants.SubLayer;
import mage.constants.SubType;
import mage.constants.Zone;
import mage.filter.common.FilterCreaturePermanent;
import mage.filter.predicate.permanent.PermanentIdPredicate;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.game.events.ZoneChangeEvent;
import mage.game.permanent.Permanent;
import mage.target.TargetCard;
import mage.target.TargetPermanent;
import mage.target.targetpointer.FixedTarget;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Auras that enchant "creature card in a graveyard" and then the creature they returned: Animate
 * Dead, Dance of the Dead, Necromancy (XMage: AnimateDeadTriggeredAbility). In a real game their
 * ETB trigger returns the creature under the Aura controller's control, adds a Custom-duration
 * layer-6 effect that swaps the enchant ability for "enchant creature put onto the battlefield
 * with this", attaches the Aura and sets up "when this leaves the battlefield, that creature's
 * controller sacrifices it". Injection fires no ETB trigger, so XMage refuses the attachment
 * (the Aura still enchants a graveyard card) and the state-based actions would bury it. This
 * rebuilds the two lasting parts of that trigger for an injected Aura and host (the host is
 * usually a control-changed permanent: Perm.owner).
 */
final class ReanimatedAura {
    private ReanimatedAura() {
    }

    /** An Aura whose enchant target is a card (in a graveyard), not a permanent or player. */
    static boolean enchantsGraveyardCard(Game game, Permanent aura) {
        if (aura == null || !aura.hasSubtype(SubType.AURA, game)) return false;
        SpellAbility sa = aura.getSpellAbility();
        return sa != null && !sa.getTargets().isEmpty() && sa.getTargets().get(0) instanceof TargetCard;
    }

    /** src: the injection's fake ability with sourceId = the Aura and controller = its controller. */
    static void install(Game game, Permanent aura, Permanent host, Ability src) {
        game.addEffect(new EnchantReturnedCreature(host.getId(), aura.getName()), src);
        game.addDelayedTriggeredAbility(new SacrificeWhenAuraLeaves(new FixedTarget(host, game)), src);
        game.applyEffects(); // so that addAttachment sees the new enchant ability
    }

    /** AnimateDeadReplaceAbilityEffect (becomesAura = false), with the returned creature fixed by id. */
    static final class EnchantReturnedCreature extends ContinuousEffectImpl {
        private final TargetPermanent newTarget;
        private final Ability newAbility;

        EnchantReturnedCreature(UUID hostId, String auraName) {
            super(Duration.Custom, Layer.AbilityAddingRemovingEffects_6, SubLayer.NA, Outcome.AddAbility);
            FilterCreaturePermanent filter = new FilterCreaturePermanent("creature put onto the battlefield with " + auraName);
            filter.add(new PermanentIdPredicate(hostId));
            newTarget = new TargetPermanent(filter);
            newAbility = new EnchantAbility(newTarget);
            staticText = "it loses \"enchant creature card in a graveyard\" and gains \"enchant creature put onto the battlefield with {this}.\"";
        }

        private EnchantReturnedCreature(final EnchantReturnedCreature effect) {
            super(effect);
            this.newTarget = effect.newTarget;
            this.newAbility = effect.newAbility;
        }

        @Override
        public EnchantReturnedCreature copy() {
            return new EnchantReturnedCreature(this);
        }

        @Override
        public boolean apply(Layer layer, SubLayer sublayer, Ability source, Game game) {
            Permanent aura = game.getPermanent(source.getSourceId());
            if (aura == null) {
                discard();
                return true;
            }
            List<Ability> toRemove = new ArrayList<>();
            for (Ability ability : aura.getAbilities(game)) {
                if (ability instanceof EnchantAbility && ability.getRule().contains(" in a graveyard")) toRemove.add(ability);
            }
            aura.removeAbilities(toRemove, source.getSourceId(), game);
            aura.addAbility(newAbility, source.getSourceId(), game);
            aura.getSpellAbility().getTargets().clear();
            aura.getSpellAbility().getTargets().add(newTarget);
            return true;
        }

        @Override
        public boolean apply(Game game, Ability source) {
            return false;
        }

        @Override
        public boolean hasLayer(Layer layer) {
            return layer == Layer.AbilityAddingRemovingEffects_6;
        }
    }

    /** AnimateDeadDelayedTriggeredAbility: when the Aura leaves the battlefield, sacrifice the creature. */
    static final class SacrificeWhenAuraLeaves extends DelayedTriggeredAbility {
        SacrificeWhenAuraLeaves(FixedTarget creature) {
            super(new SacrificeTargetEffect("that creature's controller sacrifices it"));
            setTriggerPhrase("When {this} leaves the battlefield, ");
            getEffects().setTargetPointer(creature);
        }

        private SacrificeWhenAuraLeaves(final SacrificeWhenAuraLeaves ability) {
            super(ability);
        }

        @Override
        public SacrificeWhenAuraLeaves copy() {
            return new SacrificeWhenAuraLeaves(this);
        }

        @Override
        public boolean checkEventType(GameEvent event, Game game) {
            return event.getType() == GameEvent.EventType.ZONE_CHANGE;
        }

        @Override
        public boolean checkTrigger(GameEvent event, Game game) {
            return getSourceId().equals(event.getTargetId()) && ((ZoneChangeEvent) event).getFromZone() == Zone.BATTLEFIELD;
        }
    }
}
