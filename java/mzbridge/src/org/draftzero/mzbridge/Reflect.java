package org.draftzero.mzbridge;

import java.lang.reflect.Field;

/**
 * The few engine fields the bridge has to touch without a setter (xmage_state.md G9). All of them
 * are protected/private fields of classes on the plain classpath (unnamed module), so
 * setAccessible works without --add-opens. A missing field means the jars changed under us
 * (e.g. MageZero v0.2): fail loudly with the field name instead of building a wrong state.
 */
final class Reflect {
    private Reflect() {
    }

    static Field field(Class<?> cls, String name) {
        try {
            Field f = cls.getDeclaredField(name);
            f.setAccessible(true);
            return f;
        } catch (NoSuchFieldException e) {
            throw new IllegalStateException("engine field " + cls.getName() + "." + name
                    + " not found: the XMage jars differ from the ones the bridge was written against", e);
        }
    }

    static Object get(Class<?> cls, Object o, String name) {
        try {
            return field(cls, name).get(o);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("cannot read " + cls.getSimpleName() + "." + name, e);
        }
    }

    static void set(Class<?> cls, Object o, String name, Object v) {
        try {
            field(cls, name).set(o, v);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("cannot write " + cls.getSimpleName() + "." + name, e);
        }
    }
}
