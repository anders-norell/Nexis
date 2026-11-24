# Svar på granskningen av sprint 1 (24 nov 2026)

| # | Åtgärd |
| --- | --- |
| 1 | Rättat. Frågepositioner = `n_p-1 .. L-2` (raden som väljer varje svarstoken). Bruteforce-testet räknar om med samma rättade definition, plus `mass_in_segments`, `A_first` och `A_mean`. |
| 2 | Inte rättat i sprint 1. Läckaget redovisas i tolkningen: baslinjens "A = mål" och "R = mål" kan förklaras av ordöverlappning. Kärntestet behöver uppgifter där attention och uppenbart svar skiljer sig. Det tas upp i planen för sprint 2. |
| 3 | `A_mean` (medel per token) och `A_first` (första svarssteget) sparas och redovisas bredvid A. A (summa) är fortfarande primärt. Valet låses före kärntestet. |
| 4 | `test_A_positive_control_copy`: svaret = målvärdet, kräver argmax A = mål i ≥ 36/40. Körs bara på den riktiga modellen. |
| 5 | `repetition_penalty=1.0` i `generate`. |
| 6 | `when_A_not_target`: n, R = argmax A, R = mål. |
| 7 | `transformers>=4.56`. |
| 8 | Uttrycklig systemprompt i baslinjen. Testmallen lägger nu till en standardsystemprompt som Qwen. `locate_segments` söker meddelande i i slutet av mallen renderad t.o.m. i. Nytt test: användartext som också finns i systemprompten. |
| 9 | Samtal över 1 500 tokens avvisas med ett tydligt fel (rensa chatten). |
| 10 | `answer_correct` matchar på ordgräns. |
| 11 | Nya tester: upprepad mening (`locate_given` och `locate_segments`), token som korsar meningsgräns, oberoende `mass_in_segments`, token-koppling på alla genererade uppgifter. |
| 12 | I3-mutanten ändrar första, mittersta och sista parametern. Resultatfilen sparar revision, hash för chattmallen samt versioner av torch och transformers. |
| 13 | Dokumenterat i koden att "auto" här inte betyder HF:s "auto". |
| 14 | Docstrings rättade. |
| 15 | `mean_se` hanterar n = 0. |

## Andra granskningen (`sprint1_review2.md`), åtgärder

- `locate_segments` sätter in en markör i stället för meddelandets text och letar upp markörens position.
  Korta texter som "e", "end" och "m" och upprepade meddelanden hamnar därmed rätt. Nytt test kontrollerar
  att varje segment står direkt efter sin egen rollmarkör.
- Nya tester som kan fallera: `answer_correct` med negativa fall ("maximum", "13"), `when_A_not_target` på
  syntetiska rader, `MAX_TOKENS` och att `generate` anropas med `repetition_penalty=1.0` och utan sampling.
- README beskriver den rättade A-definitionen, `A_mean` och `A_first`.
- Den positiva kontrollen körs på den riktiga modellen på produktägarens dator. Gränsen 36/40 är märkt som Claudes förslag.
- Läckaget (punkt 2) är fortfarande öppet, och det ska lösas före kärntestet.
