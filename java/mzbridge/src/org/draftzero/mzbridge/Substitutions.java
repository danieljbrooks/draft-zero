package org.draftzero.mzbridge;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import mage.cards.repository.CardInfo;
import mage.cards.repository.CardRepository;
import mage.constants.CardType;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * options.substitute: build a spec that names cards (or tokens) this XMage build does not have,
 * e.g. 2026 cube cards in an Arena log against XMage 1.4.58, by replacing them before the build.
 * The default stays strict (an unknown card fails the request).
 *
 *   {"missing": "Plains"}                  every card XMage does not know becomes a Plains
 *   {"missingToken": "SoldierToken"}       every token XMage cannot resolve becomes this class
 *   {"Sear": "Lightning Bolt", ...}        explicit replacements, applied whether or not XMage knows the card
 *
 * A name is replaced everywhere it occurs (decklist, hand, graveyard, exile, library top,
 * battlefield, stack), so the decklist accounting still holds. What a substitute cannot do is
 * dropped, with a warning: a substituted spell on the stack when the substitute is a land (a land
 * cannot be cast; its decklist copy stays in the library), and attachments, attacks, blocks and
 * stack targets that involve a substituted permanent (a Plains cannot be enchanted like the
 * creature it replaces). Every substitution is listed in the response's warnings and
 * substitutions: the built state is an approximation, never a silent one.
 */
final class Substitutions {
    private static final Map<String, Boolean> KNOWN = new ConcurrentHashMap<>();

    private Substitutions() {
    }

    static final class Result {
        /** original name -> substitute ("token X/SET" -> class for tokens) */
        final Map<String, String> applied = new TreeMap<>();
        final List<String> warnings = new ArrayList<>();

        JsonObject toJson() {
            JsonObject o = new JsonObject();
            applied.forEach(o::addProperty);
            return o;
        }
    }

    static boolean known(String name) {
        return KNOWN.computeIfAbsent(name, n -> !CardRepository.instance.findCards(n).isEmpty());
    }

    static Result apply(Spec spec, JsonObject opt) {
        Result r = new Result();
        if (opt == null) return r;
        String missing = null, missingToken = null;
        Map<String, String> explicit = new LinkedHashMap<>();
        for (Map.Entry<String, JsonElement> e : opt.entrySet()) {
            if (e.getValue() == null || e.getValue().isJsonNull()) continue;
            String v = e.getValue().getAsString();
            if (e.getKey().equals("missing")) missing = v;
            else if (e.getKey().equals("missingToken")) missingToken = v;
            else explicit.put(e.getKey(), v);
        }
        List<String> bad = new ArrayList<>();
        if (missing != null && !known(missing)) bad.add("substitute.missing '" + missing + "' is not in the XMage card database either");
        for (Map.Entry<String, String> e : explicit.entrySet()) {
            if (!known(e.getValue())) bad.add("substitute for '" + e.getKey() + "': '" + e.getValue() + "' is not in the XMage card database either");
        }
        if (missingToken != null && !tokenResolves(tokenPerm(missingToken))) bad.add("substitute.missingToken '" + missingToken + "' is not a token class");
        if (!bad.isEmpty()) throw new StateInjector.SpecException("bad options.substitute", bad);

        final String miss = missing;
        Map<String, String> memo = new HashMap<>();
        java.util.function.Function<String, String> sub = name -> {
            if (name == null) return null;
            return memo.computeIfAbsent(name, n -> {
                if (explicit.containsKey(n)) return explicit.get(n);
                return miss != null && !known(n) ? miss : n;
            });
        };
        Map<String, Map<String, Integer>> where = new TreeMap<>(); // original -> zone -> count
        java.util.function.BiFunction<List<String>, String, List<String>> zone = (cards, zoneName) -> {
            List<String> out = new ArrayList<>(cards.size());
            for (String c : cards) {
                String s = sub.apply(c);
                if (!s.equals(c)) where.computeIfAbsent(c, k -> new TreeMap<>()).merge(zoneName, 1, Integer::sum);
                out.add(s);
            }
            return out;
        };

        Set<String> gone = new HashSet<>(); // aliases of substituted permanents
        for (String seat : Spec.SEATS) {
            Spec.PlayerState p = spec.players.get(seat);
            if (p == null) continue;
            p.decklist = zone.apply(p.decklist, "decklist");
            p.hand = zone.apply(p.hand, "hand");
            p.graveyard = zone.apply(p.graveyard, "graveyard");
            p.exile = zone.apply(p.exile, "exile");
            p.libraryTop = zone.apply(p.libraryTop, "libraryTop");
            for (Spec.Perm perm : p.battlefield) {
                if (perm.isToken()) {
                    if (missingToken != null && !tokenResolves(perm)) {
                        String what = perm.describe();
                        r.applied.put(what, missingToken);
                        r.warnings.add("substitute: " + what + " -> token " + missingToken + " (XMage cannot resolve it)");
                        perm.token = null;
                        perm.set = null;
                        perm.tokenClass = missingToken;
                    }
                    continue;
                }
                String s = sub.apply(perm.name);
                if (s.equals(perm.name)) continue;
                where.computeIfAbsent(perm.name, k -> new TreeMap<>()).merge("battlefield", perm.count, Integer::sum);
                String what = seat + ":" + (perm.id == null ? perm.name : perm.id);
                perm.name = s;
                perm.set = null;
                perm.number = null;
                if (perm.id != null) {
                    gone.add(seat + ":" + perm.id);
                    for (int k = 1; perm.count > 1 && k <= perm.count; k++) gone.add(seat + ":" + perm.id + "#" + k);
                }
                if (perm.attachTo != null) {
                    r.warnings.add("substitute: dropped the attachment of " + what + " -> " + perm.attachTo);
                    perm.attachTo = null;
                }
                if (perm.counters.values().stream().anyMatch(v -> v != null && v > 0)) {
                    // counters of the original card mean nothing on the substitute
                    r.warnings.add("substitute: dropped the counters " + perm.counters + " of " + what);
                }
                perm.counters = new LinkedHashMap<>();
            }
        }
        for (String seat : Spec.SEATS) {
            Spec.PlayerState p = spec.players.get(seat);
            if (p == null) continue;
            for (Spec.Perm perm : p.battlefield) {
                if (perm.attachTo != null && gone.contains(perm.attachTo)) {
                    r.warnings.add("substitute: dropped the attachment " + seat + ":" + (perm.id == null ? perm.describe() : perm.id)
                            + " -> " + perm.attachTo + " (the host was substituted)");
                    perm.attachTo = null;
                }
            }
        }
        int attacks = spec.attackers.size(), blocks = spec.blockers.size();
        spec.attackers.removeIf(a -> gone.contains(a.attacker) || gone.contains(a.defender));
        Set<String> attacking = new HashSet<>();
        for (Spec.Attack a : spec.attackers) attacking.add(a.attacker);
        spec.blockers.removeIf(b -> gone.contains(b.blocker) || gone.contains(b.attacker) || !attacking.contains(b.attacker));
        if (spec.attackers.size() < attacks) r.warnings.add("substitute: dropped " + (attacks - spec.attackers.size()) + " attack(s) involving a substituted permanent");
        if (spec.blockers.size() < blocks) r.warnings.add("substitute: dropped " + (blocks - spec.blockers.size()) + " block(s) involving a substituted permanent");

        List<Spec.StackItem> stack = new ArrayList<>();
        for (Spec.StackItem si : spec.stack) {
            String s = sub.apply(si.card);
            if (!s.equals(si.card)) {
                where.computeIfAbsent(si.card, k -> new TreeMap<>()).merge("stack", 1, Integer::sum);
                if (isLand(s)) {
                    r.warnings.add("substitute: dropped " + si.controller + "'s " + si.card + " from the stack (a " + s
                            + " cannot be cast; that copy stays in the library)");
                    continue;
                }
                si.card = s;
            }
            for (int i = 0; i < si.targets.size(); i++) {
                if (gone.contains(si.targets.get(i))) {
                    // later targets would slide into the wrong slot: stop at the first lost one
                    r.warnings.add("substitute: dropped " + (si.targets.size() - i) + " target(s) of " + si.card + " from " + si.targets.get(i));
                    si.targets = new ArrayList<>(si.targets.subList(0, i));
                    break;
                }
            }
            stack.add(si);
        }
        spec.stack = stack;

        List<String> summary = new ArrayList<>();
        for (Map.Entry<String, Map<String, Integer>> e : where.entrySet()) {
            String s = sub.apply(e.getKey());
            r.applied.put(e.getKey(), s);
            boolean unknown = !explicit.containsKey(e.getKey());
            summary.add("substitute: '" + e.getKey() + "' -> '" + s + "'" + (unknown ? " (not in the XMage card database)" : "")
                    + ": " + e.getValue());
        }
        r.warnings.addAll(0, summary);
        return r;
    }

    static boolean isLand(String name) {
        CardInfo ci = StateInjector.cardInfo(name, null, null);
        return ci.getTypes().contains(CardType.LAND);
    }

    static Spec.Perm tokenPerm(String cls) {
        Spec.Perm p = new Spec.Perm();
        p.tokenClass = cls;
        return p;
    }

    /** The token names exactly one loadable class (an ambiguous name is not "missing": it fails the build). */
    static boolean tokenResolves(Spec.Perm perm) {
        try {
            Class.forName(StateInjector.tokenClassName(perm));
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        } catch (StateInjector.SpecException e) {
            return !e.getMessage().startsWith("unknown token");
        }
    }
}
