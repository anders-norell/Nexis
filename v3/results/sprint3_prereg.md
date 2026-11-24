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

## Tillägg 2b/2c (24 nov 2026 ca 23:20, efter steg 2:s underkännande, före all 2b-träning)

Anders: "now we are talking. invarianter" (24 nov 22:57) på förslaget om symmetri. Allt nedan är [Claudes förslag].

**Steg 2 underkänt** (`sprint3_schema.json`): schemat lärde sig ämnena. Testtexterna test3 är därmed sedda och används
bara beskrivande.

**2b, relationsschema** (`nexis3/schema_rel.py`, `train_schema3b.py train`): ser bara skalära relationer mellan
tillstånden (cosinus mot sista prompttoken, inlärd lågrangslikhet med gemensamt W, likhet mot övriga meningar, centrerat
inom texten) och meningens position, aldrig tillståndsvektorerna själva. Rutnät k ∈ {0, 8, 16, 32}. Validering på
ämnesnivå: sex träningsämnen undanhållna (frö 31); träning = texter utan dem, validering = texter vars målämne är ett av
dem; val bara på valideringens KL. Arkitektur och rutnät bestämda nu och ändras inte efter att test3-utfallet setts.
Beskrivande: resultat på test3 och karikatyrmåttet (Spearman mellan parvisa avstånd i A och S; överlapp bland de 10 %
största avstånden), jämfört med steg 2:s schema.

**Låsning:** efter 2b skrivs `sprint3_schema3b_lock.json` med schemats sha256 innan test3b finns i arbetsytan. Inget
schema byts efter det.

**2c, nya testtexter** (`extract.py --split test3b`, frö 7000): 300 symmetripar på de tio testämnena. Tvillingen har samma
målposition, lockposition och fyllnadspositioner men tre andra ämnen (inget gemensamt ämne inom paret).

**Kriterier 2c (låsta här, en körning):**
1. *Huvud* (samma som steg 2): på originalen (300), argmax S = argmax A parat mot "alltid målmeningen", skillnad > 2 SE.
2. *Symmetri*: per par dS = L1(S, S′) och dA = L1(A, A′). Godkänt om schemat inte ändras signifikant mer än
   uppmärksamheten: medel(dS − dA) < 2 SE. (Ett schema som följer ämnet ändras mer än A; ett som följer A:s roller inte.)
Steg 2c klarat = båda. Klaras det inte stannar vi och diskuterar; nästa väg är då ordbokssporet (vad som är i fokus).
Redovisas också: tvillingarnas huvudmått, sista mening, ordöverlapp, S när A ≠ mål, karikatyrmåttet.
