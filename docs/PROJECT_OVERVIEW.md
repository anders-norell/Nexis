# Nexis — projektöversikt

Läs den här filen först. Den beskriver var allt finns, vad som är gjort och hur man tar upp arbetet igen.

**Produktägare:** Anders. **Byggt av:** Claude, oktober 2026.
**Status 7 okt 2026:** pausat efter v2. Nästa steg ej beslutat.

## Syfte

En liten proof of concept av Grazianos Attention Schema Theory (AST). En konstgjord agent följer ett mål med blicken och har en inre modell (ett "schema") av sin egen uppmärksamhet. Huvudfrågan: när agenten rapporterar var den tittar utifrån schemat, gör den då de systematiska fel AST förutsäger?

Kopplingen till MACTA: om det fungerar blir kriteriet *Awareness* mätbart i stället för bara definierat.

## Var allt finns

| Vad | Var |
| --- | --- |
| Spec, hypoteser, trösklar och alla resultat | `claude/nexis_spec.md` i projektet (ögonblicksbild 7 okt). Levande version med arkitekturdiagram: Claude-dokumentet "Nexis PoC – Attention Schema Specification" |
| All källkod, läsbar | `nexis_source.md` |
| Kod + tränade modeller + rådata | `nexis_v2.zip` |
| Figur från Sprint 1 | `report/sprint1.png` i zip-filen |

## Vad som är gjort

| Steg | Resultat |
| --- | --- |
| Sprint 1: miljö och agent utan schema | Klar. Styrd uppmärksamhet ger 66–68 % rätt pixel, mot 39 % med jämn uppmärksamhet och 1,6 % slump. |
| Sprint 2 (v1): schema tillagt | H0 och H1 föll. Schemat hjälper, men bara 2–3 procentenheter. Krav var 10 och 20. |
| v2: drift i uppmärksamheten + grov spårsignal | H0, H0b och H1 föll. Agenter utan schema klarar sig lika bra. |
| Explorativ observation (v2) | När uppmärksamheten drev iväg rapporterade schemat ofta den *avsedda* platsen (31 %, mot cirka 7 % av slump). Det liknar AST:s förutsägelse, men är inte förregistrerat. |

## Beslut som gäller

- Trösklar sätts **innan** körning och sänks aldrig i efterhand.
- Stoppregel: om schemat inte hjälper styrningen (H0/H1) körs inte H2/H3 utan nytt beslut.
- Inga fler försök att "få H0 att gå igenom" utan ny förregistrering. Risk för att leta tills något råkar fungera.

## Öppna vägar för nästa försök

1. Skriva ihop resultaten som de är och kontakta Grazianos labb.
2. Förregistrera ett test av den explorativa observationen (H2) på befintliga v2-modeller. Bryter mot stoppregeln, vilket då måste redovisas öppet.
3. Ny uppgift där agenten verkligen behöver känna sin egen uppmärksamhet, närmare Wilterson & Graziano 2021. Kräver ny spec.

## Köra koden

Kräver bara Python 3, NumPy och Matplotlib (PyTorch användes inte; en egen liten autodiff ersätter den).

```
unzip nexis_v2.zip && cd nexis
export OMP_NUM_THREADS=1
python3 test_autodiff.py     # gradientkontroll
./run_v2.sh                  # 15 träningskörningar, ca 30 min på 2 kärnor
python3 eval_v2.py           # domar för H0, H0b, H1
```

## Viktiga designval (detaljer i specen)

- 8×8 synfält, 16 rutor, målet driver i 12 steg. Agenten ska ange exakt pixel.
- Uppmärksamheten följer agentens egen tro om var målet är, och begränsas till ±4 så att en blixt (+10) alltid vinner.
- Styrenheten ser inte blixten. I v2 driver uppmärksamheten slumpmässigt (AR(1), SD 2,5), och schemat får en brusig signal per kvadrant.
- Kontroller med lika många parametrar (C, C2) för att utesluta att schemat bara hjälper för att nätverket blir större.
