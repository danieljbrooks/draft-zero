package org.draftzero.mzbridge;

import java.security.Provider;
import java.security.SecureRandom;
import java.security.SecureRandomSpi;
import java.security.Security;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.UUID;

/**
 * Makes {@link UUID#randomUUID()} reproducible inside a request.
 *
 * XMage gives every card, permanent and ability a random UUID, and the engine orders by them in
 * places that change play: attack and block questions are asked in UUID order
 * (ComputerPlayer.selectAttackersOneAtATime), MCTS children are sorted by a string that contains
 * source/target UUIDs (MCTSNode.getOrderString), and HashSet iteration over ids decides target
 * order. So "same spec + same seed" only gives the same game, the same decision and the same
 * search statistics if the ids are the same too.
 *
 * UUID.randomUUID() draws from a SecureRandom created with the default provider on first use.
 * {@link #install()} registers a provider first in the list before anything asks for a UUID; its
 * generator delegates to the platform's NativePRNG unless {@link #reset(long)} has put it in
 * seeded mode, where bytes come from a SplittableRandom. Only the worker process does this (the
 * fork is unchanged), and the seeded stream is for ids only: nothing security-relevant runs in a
 * worker.
 */
final class DeterministicIds {
    private static volatile SplittableRandom stream;
    private static final Object LOCK = new Object();
    private static final SecureRandom REAL;
    private static boolean installed;

    static {
        SecureRandom r;
        try {
            r = SecureRandom.getInstance("NativePRNG", "SUN");
        } catch (Exception e) {
            r = new SecureRandom(); // called before install(), so this is still the platform default
        }
        REAL = r;
    }

    private DeterministicIds() {
    }

    /** Must run before the first UUID.randomUUID() in the JVM. Returns false if that already happened. */
    static synchronized boolean install() {
        if (installed) return true;
        Provider p = new Provider("MzBridgeIds", "1.0", "reproducible UUIDs for the mzbridge worker") {
            {
                putService(new Provider.Service(this, "SecureRandom", "MzBridgeIds", Spi.class.getName(),
                        null, Map.of("ThreadSafe", "true")));
            }
        };
        Security.insertProviderAt(p, 1);
        reset(1L);
        UUID a = UUID.randomUUID();
        reset(1L);
        UUID b = UUID.randomUUID();
        off();
        installed = a.equals(b);
        return installed;
    }

    static boolean isInstalled() {
        return installed;
    }

    /** Seeded mode: every following UUID is a function of the seed and of the order of requests. */
    static void reset(long seed) {
        synchronized (LOCK) {
            stream = new SplittableRandom(seed ^ 0x6A09E667F3BCC909L);
        }
    }

    static void off() {
        synchronized (LOCK) {
            stream = null;
        }
    }

    public static final class Spi extends SecureRandomSpi {
        public Spi() {
        }

        @Override
        protected void engineSetSeed(byte[] seed) {
        }

        @Override
        protected void engineNextBytes(byte[] bytes) {
            synchronized (LOCK) {
                SplittableRandom s = stream;
                if (s == null) {
                    REAL.nextBytes(bytes);
                    return;
                }
                for (int i = 0; i < bytes.length; i++) bytes[i] = (byte) s.nextInt(256);
            }
        }

        @Override
        protected byte[] engineGenerateSeed(int numBytes) {
            return REAL.generateSeed(numBytes);
        }
    }
}
