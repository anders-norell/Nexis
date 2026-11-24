# Sprint 3 – förregistrering (24 nov 2026, före all data för sprint 3)

Plan: projektet `claude/nexis_v3_sprint3_plan.md`, godkänd av Anders 24 nov 18:30 (beslut 2b, styrning v2 i sprint 4).

**Ämnen:** 40 (`nexis3/topics_v3.py`). Testämnen, valda innan någon data fanns: boat, chess, bike, lang, film, shoe,
hotel, dish, river, planet. Testtexter har bara testämnen (alla fyra meningarna). Träningstexter har bara träningsämnen.

**Data:** träning frö 5000, 1 500 texter (sista 200 = validering för tidig stopp och hyperparametrar). Test frö 6000,
300 texter. Samma A-mått och samma tillstånd (lager 9/13/17) som i sprint 2.

**Steg 2-kriterium (schemat på okända ämnen):** på testtexterna, andel där argmax S = argmax A, parat mot nollmodellen
"alltid målmeningen" (argmax A = mål). Godkänt om skillnaden > 2 SE. Redovisas också mot "sista meningen" och
ordöverlapp. Klaras det inte stannar vi och diskuterar med Anders innan rapportmodellen byggs.

**Grind 3-kriterium för rapporten** låses i ett tillägg innan LoRA tränas.
