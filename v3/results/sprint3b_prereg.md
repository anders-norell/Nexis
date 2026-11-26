# Sprint 3b – förregistrering (26 nov 2026 ca 07:10, före all träning; testtexterna ej hämtade till arbetsytan)

Plan: projektet `claude/nexis_v3_sprint3b_plan.md`, godkänd av Anders 25 nov 23:45 ("vi kör på det").
Tekniska val nedan är [Claudes förslag].

## Data
- Träning: `tok_train` (frö 5000, 1 500 texter, träningsämnen). Validering: texter vars målämne hör till sex
  undanhållna träningsämnen (frö 31, samma regel som 2b); träning = texter utan något av de sex.
- Test: `tok_test` (frö 8000, 300 nya texter, de tio testämnena). Hämtas till arbetsytan först efter låsningen.
- A_t = attention från raden som väljer svarstoken t, per lager (28), huvud (12) och textdel (mening 1–4, fråga, övrigt,
  svaret hittills) = 2 352 tal, standardiserade med träningens medel och spridning.

## Modell (`nexis3/compress.py`)
- P: 2 352 → 64 → k (k = 4 huvudfall; 2 och 8 beskrivande). Ser bara A_t.
- G: S_{t+1} ≈ S_t + MLP(S_t, q8), q8 = linjär projektion av frågans läge (lager 13) till 8 tal.
- Y-huvud: [medel över svarets token av S, S vid första token] → rätt/fel.
- Förlust: ε + λ·Y-förlust, ε = E‖S_{t+1} − Ŝ‖² / E‖S − S̄‖² (skalinvariant).
- λ ∈ {0,1; 1; 10}. Valregel: bland λ med ε_G < ε_identitet på validering, lägst Y-förlust på validering; annars lägst ε_G.
  k = 2 och 8 tränas med samma λ. Ett träningsfrö (0).

## Kriterier för grind 3 (en körning på test, k = 4)
- **K1 kommutering:** per text medel av ε_identitet − ε_G (samma nämnare för båda, från testets S), > 2 SE över texter
  med minst en övergång.
- **K2 nytta:** nya Y-huvuden (samma arkitektur och träning: tränas på träningsdelen, tidigt stopp på valideringen) på
  fryst S respektive den handvalda karikatyren (andel per mening i lager 9–17, per token, samma poolning). Per text
  log-förlust(karikatyr) − log-förlust(S) > 2 SE.
- **K3 trohet:** Spearman mellan parvisa avstånd i A (standardiserad) och i S, minus samma för en fast slumpprojektion
  till k tal (frö 12345). Bootstrap över texter (200 dragningar, 400 token per dragning): medel > 2 × bootstrapspridning.
- Grind 3 klarad = K1 och K2 och K3.
- Redovisas också: k = 2 och 8, Y-förlust för konstant prediktion, korrelation mellan S:s tal och andelarna på
  målmening, lockmening och varje mening, ε per övergång.

## Kända begränsningar (före utfall)
- Svaren är korta (median 3 token), så dynamiken bygger på få övergångar per text.
- Bi-Y (vilken mening svaret hämtas ur) mäts med ordöverlapp och är brusigt: lockmeningen delar ofta ord med svaret
  (i träningsdatan "lock" 481 av 1 500). Därför bara beskrivande.
- Uppgiften rätt i 75 % av träningstexterna.
