import { useMemo, useState } from 'react';
import type { Relationship } from '../api.js';

interface Props {
  npcId: string;
  npcName: string;
  relationships: Relationship[];
  onSelectNPC?: (id: string) => void;
}

interface NodePos {
  x: number;
  y: number;
  rel: Relationship;
}

const CX = 200;
const CY = 200;
const RADIUS = 140;
const NODE_R = 22;
const CENTER_R = 28;

function relationshipColor(rel: Relationship): string {
  if (rel.conflict > 0.5) return '#c85c5c';
  if (rel.conflict > 0.3) return '#c8a84b';
  const score = (rel.affection + rel.trust) / 2;
  if (score > 0.6) return '#5c8c6c';
  if (score > 0.3) return '#6c8ccc';
  return '#6e6e7a';
}

function lineOpacity(rel: Relationship): number {
  const strength = Math.max(rel.familiarity, rel.conflict, (rel.affection + rel.trust) / 2);
  return 0.3 + strength * 0.7;
}

function truncate(name: string, max = 10): string {
  return name.length > max ? name.slice(0, max - 1) + '…' : name;
}

export function RelationshipGraph({ npcId: _npcId, npcName, relationships, onSelectNPC }: Props) {
  const [hovered, setHovered] = useState<string | null>(null);

  const nodes = useMemo<NodePos[]>(() => {
    const count = relationships.length;
    if (count === 0) return [];
    return relationships.map((rel, i) => {
      const angle = (2 * Math.PI * i) / count - Math.PI / 2;
      return {
        x: CX + RADIUS * Math.cos(angle),
        y: CY + RADIUS * Math.sin(angle),
        rel,
      };
    });
  }, [relationships]);

  const hoveredNode = hovered ? nodes.find(n => n.rel.targetId === hovered) : null;

  if (relationships.length === 0) {
    return <div className="empty">No relationships to graph yet.</div>;
  }

  return (
    <div className="rel-graph-wrap">
      <svg
        viewBox="0 0 400 400"
        className="rel-graph-svg"
        aria-label={`Relationship graph for ${npcName}`}
      >
        {/* Lines */}
        {nodes.map(({ x, y, rel }) => (
          <line
            key={`line-${rel.targetId}`}
            x1={CX} y1={CY}
            x2={x} y2={y}
            stroke={relationshipColor(rel)}
            strokeWidth={1.5 + rel.familiarity * 2}
            strokeOpacity={lineOpacity(rel)}
          />
        ))}

        {/* Outer nodes */}
        {nodes.map(({ x, y, rel }) => {
          const color = relationshipColor(rel);
          const isHovered = hovered === rel.targetId;
          return (
            <g
              key={rel.targetId}
              onClick={() => onSelectNPC?.(rel.targetId)}
              onMouseEnter={() => setHovered(rel.targetId)}
              onMouseLeave={() => setHovered(null)}
              onFocus={() => setHovered(rel.targetId)}
              onBlur={() => setHovered(null)}
              style={{ cursor: onSelectNPC ? 'pointer' : 'default' }}
              tabIndex={onSelectNPC ? 0 : -1}
              role={onSelectNPC ? 'button' : undefined}
              aria-label={`${rel.targetName} — ${rel.kinshipType}`}
            >
              <circle
                cx={x} cy={y} r={isHovered ? NODE_R + 3 : NODE_R}
                fill="#1a1a24"
                stroke={color}
                strokeWidth={isHovered ? 2.5 : 1.5}
              />
              <text
                x={x} y={y + 4}
                textAnchor="middle"
                fontSize={9}
                fill={color}
                fontFamily="inherit"
              >
                {truncate(rel.targetName, 9)}
              </text>
            </g>
          );
        })}

        {/* Center node */}
        <circle cx={CX} cy={CY} r={CENTER_R} fill="#1a1a24" stroke="#c8a84b" strokeWidth={2} />
        <text x={CX} y={CY + 4} textAnchor="middle" fontSize={9} fill="#c8a84b" fontFamily="inherit">
          {truncate(npcName, 10)}
        </text>
      </svg>

      {/* Tooltip */}
      {hoveredNode && (
        <div className="rel-graph-tooltip">
          <strong>{hoveredNode.rel.targetName}</strong>
          <span className="rel-type-badge">{hoveredNode.rel.kinshipType}</span>
          <div className="rel-graph-stats">
            <span>Affection {(hoveredNode.rel.affection * 100).toFixed(0)}%</span>
            <span>Trust {(hoveredNode.rel.trust * 100).toFixed(0)}%</span>
            <span>Familiarity {(hoveredNode.rel.familiarity * 100).toFixed(0)}%</span>
            {hoveredNode.rel.conflict > 0.3 && (
              <span className="conflict-flag">Conflict {(hoveredNode.rel.conflict * 100).toFixed(0)}%</span>
            )}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="rel-graph-legend">
        <span><span className="legend-dot" style={{ background: '#5c8c6c' }} /> Strong bond</span>
        <span><span className="legend-dot" style={{ background: '#6c8ccc' }} /> Acquaintance</span>
        <span><span className="legend-dot" style={{ background: '#c8a84b' }} /> Tension</span>
        <span><span className="legend-dot" style={{ background: '#c85c5c' }} /> Conflict</span>
      </div>
    </div>
  );
}
