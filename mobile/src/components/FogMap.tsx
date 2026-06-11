import React from "react";
import { ScrollView, Text, View } from "react-native";
import { parseCell } from "../location";
import { styles } from "../theme";

/**
 * Fog-of-war territory map: every tile is a ~400m cell the host has actually
 * stood in. Rendered entirely from cell ids — no map SDK, no API key, no
 * coordinates. Positions are relative to the bounding box of discovered
 * territory, so the map shows shape and growth, not real-world geography.
 *
 * Render cap: the most recent MAX_TILES cells. A long-lived save can hold
 * thousands of cells; capping keeps the World tab fast while still showing
 * the living frontier (newest territory is what players care about).
 */
const TILE = 12;
const GAP = 2;
const MAX_TILES = 400;

export function FogMap({ cells, currentCell }: { cells: string[]; currentCell?: string | null }) {
  // Always include the current cell so "you are here" renders even when it
  // falls outside the recency window.
  const withCurrent = currentCell && !cells.includes(currentCell) ? [...cells, currentCell] : cells;
  const recent = withCurrent.slice(-MAX_TILES);
  const parsed = recent
    .map((id, index) => ({ ...parseCell(id), order: index, here: id === currentCell }))
    .filter((p): p is { row: number; col: number; order: number; here: boolean } => p.row !== undefined && p.col !== undefined);

  if (!parsed.length) {
    return (
      <Text style={styles.muted}>
        No territory charted yet. Turn on location quests in Settings, then move through the world — areas you visit
        appear here.
      </Text>
    );
  }

  const minRow = Math.min(...parsed.map((p) => p.row));
  const maxRow = Math.max(...parsed.map((p) => p.row));
  const minCol = Math.min(...parsed.map((p) => p.col));
  const maxCol = Math.max(...parsed.map((p) => p.col));
  const width = (maxCol - minCol + 1) * (TILE + GAP);
  const height = (maxRow - minRow + 1) * (TILE + GAP);
  const latestOrder = parsed.length - 1;

  return (
    <View>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        <ScrollView showsVerticalScrollIndicator={false} style={{ maxHeight: 260 }}>
          <View style={{ width, height: Math.min(height, 2000) }}>
            {parsed.map((p) => {
              const isLatest = p.order === latestOrder;
              const isHere = p.here;
              // Newer cells glow brighter: the frontier reads as alive.
              const recency = 0.35 + 0.65 * (p.order / Math.max(1, latestOrder));
              return (
                <View
                  key={`${p.row}_${p.col}`}
                  style={{
                    position: "absolute",
                    left: (p.col - minCol) * (TILE + GAP),
                    // Higher latitude row = further north = visually higher.
                    top: (maxRow - p.row) * (TILE + GAP),
                    width: TILE,
                    height: TILE,
                    borderRadius: 2,
                    backgroundColor: isHere ? "#fbbf24" : isLatest ? "#67e8f9" : `rgba(56, 189, 248, ${recency.toFixed(2)})`,
                    borderWidth: isHere ? 1 : 0,
                    borderColor: "#fef3c7"
                  }}
                />
              );
            })}
          </View>
        </ScrollView>
      </ScrollView>
      <Text style={styles.muted}>
        {cells.length} area{cells.length === 1 ? "" : "s"} charted · each tile ≈ 400m · bright = recent{currentCell ? " · gold = you" : ""}
      </Text>
    </View>
  );
}
