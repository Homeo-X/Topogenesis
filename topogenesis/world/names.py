from __future__ import annotations

GIVEN_NAMES: tuple[str, ...] = (
    "Aelric",
    "Aldwin",
    "Anwen",
    "Brenna",
    "Cael",
    "Cerys",
    "Dain",
    "Eira",
    "Elowen",
    "Fenn",
    "Garric",
    "Hale",
    "Ilyra",
    "Jorin",
    "Kael",
    "Lysa",
    "Maera",
    "Neris",
    "Oren",
    "Perrin",
    "Rowan",
    "Sable",
    "Tavia",
    "Ulric",
    "Veyra",
    "Wren",
    "Ysold",
    "Zerin",
)

FAMILY_NAMES: tuple[str, ...] = (
    "Ashbourne",
    "Blackfen",
    "Briarholt",
    "Duskmere",
    "Elderbrook",
    "Emberlain",
    "Fallowmere",
    "Grayward",
    "Harthorne",
    "Ironvale",
    "Lowenfield",
    "Mosswick",
    "Redwillow",
    "Reedwatch",
    "Stonewell",
    "Thornhollow",
    "Valehart",
    "Wintermere",
)

CALLINGS: tuple[str, ...] = (
    "apothecary",
    "beekeeper",
    "blacksmith",
    "charcoal burner",
    "field hand",
    "forager",
    "herbalist",
    "miller",
    "reed-cutter",
    "shepherd",
    "stone mason",
    "tanner",
    "weaver",
    "woodward",
)


def villager_display_name(npc_id: int, lineage: int = 0, generation: int = 0) -> str:
    """Return a stable, human-readable village name for an NPC."""
    seed = int(npc_id) * 1_103 + int(lineage) * 97 + int(generation) * 41
    given = GIVEN_NAMES[seed % len(GIVEN_NAMES)]
    family = FAMILY_NAMES[(seed // len(GIVEN_NAMES)) % len(FAMILY_NAMES)]
    return f"{given} {family}"


def villager_calling(npc_id: int, lineage: int = 0, generation: int = 0) -> str:
    seed = int(npc_id) * 313 + int(lineage) * 17 + int(generation) * 7
    return CALLINGS[seed % len(CALLINGS)]
