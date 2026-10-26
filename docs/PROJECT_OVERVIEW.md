# Nexis — projektöversikt

Uppdaterad 26 okt 2026. Läs den här filen först om du är ny i repot.

**Produktägare:** Anders Norell. **Byggt av:** Claude (Anthropic), oktober 2026. All kod är skriven av en AI och har
granskats av separata AI-instanser, men ännu inte av en människa med ML-utbildning.

## Vad projektet prövar

Nexis prövar Michael Grazianos Attention Schema Theory (AST) i små artificiella agenter: kommer en agents kunskap om
sin egen uppmärksamhet från en förenklad modell av uppmärksamheten (ett schema), och syns det i hur schemat har fel?

## Två försök

| | v1 (avslutat) | v2 (avslutat 26 okt, i extern granskning) |
| --- | --- | --- |
| Var | rotkatalogen (`*.py`, `results/`, `report/`) | `v2/` |
| Spec | `docs/spec.md` (7 okt) | levande dokument + `docs/spec_v2.md` (ögonblicksbild 23 okt) |
| Uppgift | följa ett mål med mjuk uppmärksamhet, övervakad inlärning | söka i rutnät med hård spotlight som kan kapas, förstärkningsinlärning |
| Resultat | schemat hjälpte lite; bifynd: fel drog mot avsedd plats (ej förregistrerat) | se nedan |

## v2: vad vi kom fram till

Rapport (engelska): `docs/report_v2.md`, levande version https://claude.ai/artifact/WgWqgRBXEqaChu3NMB76JF.

1. Ett korrekt schema som ges utifrån hjälper mycket (+6–7,5 procentenheter) och agenten lyder det även när det ljuger
   (H2, förregistrerat, klarat). Det visar att schemat behandlas som auktoritet, inte att det är förenklat.
2. Ett schema som agenten fick lära sig själv (1b med facit, 1c utan) gav ingen självmodell. Den lilla vinsten kom från
   en handkodad felsignal ("det jag såg stämmer inte med var jag siktade"); samma signal direkt till agenten utan modul
   gav samma vinst (steg 0, förregistrerat).
3. AST:s kännetecken, att schemats fel drar mot den avsedda platsen (H5), syntes inte.
4. Metodlärdom: mått på "pekar schemat på avsikten" blev fem gånger sanna eller falska enbart genom sin konstruktion.

## Var saker finns i v2

| Vad | Var |
| --- | --- |
| Börja här | `v2/README.md` |
| Återskapa tabellerna | `python3 v2/reproduce.py tables` (sekunder) eller `evaluate` (ca 35 min) |
| Resultat, förregistreringar, låsfiler, granskningar | `v2/results/` |
| Arbetsdagbok sprint för sprint | `CLAUDE.md`, avsnittet Status |
| Begrepp i vardagsspråk | https://claude.ai/artifact/2syuVpzf2GcAfaP9DgSgjC (ordlista) |

## Status och nästa steg

Beslut 26 okt: ingen ny studie innan rapporten granskats av en människa med ML-erfarenhet. Därefter kontakt med
Grazianos grupp och Piefke m.fl. Sedan beslut om nästa studie (social tes, inre störning eller ingen), med egen spec.
