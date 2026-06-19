import { describe, it, expect } from 'vitest';
import { createRelationship, applyInteraction, driftRelationship } from '../src/npc/relationships.js';

describe('Relationship system', () => {
  it('creates a stranger relationship with low familiarity', () => {
    const rel = createRelationship('npc2', 'stranger');
    expect(rel.familiarity).toBe(0);
    expect(rel.kinshipType).toBe('stranger');
  });

  it('gifts increase affection and trust', () => {
    const rel = createRelationship('npc2', 'acquaintance');
    const updated = applyInteraction(rel, 'gift', 5, 0.8);
    expect(updated.affection).toBeGreaterThan(rel.affection);
    expect(updated.trust).toBeGreaterThan(rel.trust);
  });

  it('betrayal decreases trust significantly', () => {
    const rel = createRelationship('npc2', 'friend');
    const updated = applyInteraction(rel, 'betrayal', 5, 1.0);
    expect(updated.trust).toBeLessThan(rel.trust);
    expect(updated.conflict).toBeGreaterThan(rel.conflict);
  });

  it('arguments increase conflict', () => {
    const rel = createRelationship('npc2', 'acquaintance');
    const updated = applyInteraction(rel, 'argument', 5);
    expect(updated.conflict).toBeGreaterThan(rel.conflict);
  });

  it('relationships drift toward neutral if not interacted with', () => {
    const rel = { ...createRelationship('npc2', 'friend'), affection: 0.8, familiarity: 0.9, lastInteractionDay: 0 };
    const drifted = driftRelationship(rel, 30);
    expect(drifted.familiarity).toBeLessThan(rel.familiarity);
  });

  it('kin relationships start with higher affection', () => {
    const kinRel = createRelationship('npc2', 'kin');
    const strangerRel = createRelationship('npc3', 'stranger');
    expect(kinRel.affection).toBeGreaterThan(strangerRel.affection);
    expect(kinRel.trust).toBeGreaterThan(strangerRel.trust);
  });
});
