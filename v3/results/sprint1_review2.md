# Andra granskningen av Nexis v3, sprint 1 (kontroll av åtgärderna)

Granskare: fristående Claude-instans, 24 nov 2026. Underlag: `results/sprint1_review.md`,
`results/sprint1_review_response.md` och koden i `v3/` som den ligger nu. Inga befintliga filer har ändrats,
inget har pushats. Egna prov ligger under `/tmp/rev2/` (`mut.py`, `edge.py`).

**Testsviten:** `python3 -m pytest -q` ger **49 passed, 1 skipped** (3,8 s, testmodellen, transformers 5.24).
Det överhoppade testet är den positiva kontrollen, som bara körs på den riktiga modellen.

## Dom

**Godkänd med anmärkningar.** Alla 15 åtgärder finns i koden. Punkt 1 är korrekt rättad och bruteforce-testet
fångar nu både den gamla och andra förskjutningar. Punkt 8 löser fallet med Qwens automatiska systemprompt,
men den nya sökningen har ett eget tyst fel: korta meddelanden som också finns i mallens avslutande markör
placeras i markören (anmärkning 1 nedan). Det rör bara chatten. Några åtgärder saknar test som kan fallera.

---

## Punkt 1: frågepositionerna

**Koden.** `attention.py` r. 183: `q = list(range(n_p - 1, len(full) - 1)) if answer_ids else [n_p - 1]`.
Rad *t* förutsäger token *t+1*. Rad `n_p − 1` (sista prompttoken, `\n` efter `assistant`) väljer svarets första
token, och rad `L − 2` väljer den sista. Raden som väljer EOS är borta. `generate` tar bort avslutande EOS
(r. 168–169), så `answer_ids` innehåller ingen EOS och intervallet blir rätt även för genererade svar.
Fallbacken utan svar är nu samma rad som det första steget i huvudfallet, och då är `A_first == A`
(kontrollerat i `edge.py`). Docstringen (r. 3–15) beskriver definitionen rätt. `A_first` = `seg_mass[:, 0, :]`
= rad `n_p − 1`. Det stämmer.

**Bruteforce-testet** (`test_I1_matches_bruteforce`). Det räknar om med egna loopar: lager × huvud × q × k,
`qs = range(len(ids) − 1, len(ids) + len(ans) − 1)`. Det skriver alltså intervallet på nytt, i stället för att
läsa det ur produktionskoden. Det kör modellen självt (utan attention_mask, vilket är ekvivalent utan padding).
`mass_in_segments` räknas nu oberoende: nämnaren `allmass` summerar alla vikter på raderna och tar inte
produktionens `total`. Det kontrollerar också `A_first` och `A_mean`.
Oberoendet har en gräns. Testet återanvänder `measurer.encode`, `token_segment_ids`, `m.segments` och samma
fristående tokenisering av svaret. Det prövar alltså summeringen och valet av rader, men inte segmentkopplingen.
Den prövas i I2. Testet kodar också samma definition, så det kan inte fånga ett fel i själva definitionen.
Definitionen är nu korrekt, så det gör inget.

**Mutationsprov** (`/tmp/rev2/mut.py`, hela sviten per mutant):

| Mutant | Utfall | Avvikelse mot toleransen 1e-5 |
| --- | --- | --- |
| gamla `range(n_p, L)` | dödad (bruteforce) | 3,0e-4 |
| båda ändarna `range(n_p − 1, L)` | dödad | 1,6e-4 |
| bara första steget | dödad | 5,1e-4 |
| `A_first` från sista raden | dödad | 1,9e-3 |
| `mass_in_segments` = 1,0 (överlevde förra gången) | **dödad** | – |
| `A_mean` utan division med antal tokens | dödad | 5,2e-2 |

Marginalen är minst 16 gånger toleransen. Punkt 1 är **genomförd och korrekt**.

## Punkt 8: `locate_segments` och den automatiska systemprompten

**Koden.** `attention.py` r. 114–137. För varje meddelande *i* renderas mallen till och med *i*
(`add_generation_prompt=False`). Koden kräver att det är ett prefix av prompten, annars ValueError, vilket
är ett högljutt fel och inte tyst. Sedan tar den `pre.rfind(content)`.
Testmallen i `conftest.py` lägger nu till en standardsystemprompt när ingen ges, som Qwen2.5 gör.
`baseline.py` r. 60 anger en uttrycklig systemprompt (`SYSTEM_PROMPT`), och den sparas i resultatfilen.

**Test.** `test_I2_not_fooled_by_default_system_prompt` använder som användartext den första meningen i
standardsystemprompten. Testet kräver positionen för den sista förekomsten och att den skiljer sig från den
första. Det dödar den gamla sökningen från position 0 och `pre.find`. Med den riktiga Qwen-mallen ger
`prompt0.split("\n")[1]` systempromptens rad, så testet fungerar även där.

**Nytt tyst fel (anmärkning 1).** `rfind` i `pre` hittar den *sista* förekomsten. `pre` slutar med
`content + "<|im_end|>\n"`. Ett meddelande som är en delsträng av den avslutande markören placeras därför
i markören. Det gäller till exempel "e", "m", "d", "n", "end", "im" och "|". Provat med testmallen:

| Meddelande | Segmentet hamnar på |
| --- | --- |
| `e` | tecknet "e" i `<|im_end|>` |
| `end` | "end" i `<|im_end|>` |
| `m` | "m" i `<|im_end|>` |

`measure` går igenom utan fel och ger A = [1,0]. Mätningen gäller då markörens token i stället för meddelandet.
Felet rör bara chatten, och bara meddelanden på några tecken. Det är samma typ av fel som punkt 8 skulle
ta bort. **Förslag:** sök i fönstret som meddelandet lägger till, alltså
`pre.find(content, len(render(messages[:i])))` (för i = 0 krävs att mallen tål en tom lista). Alternativt kan
meddelandet renderas med en unik platshållare, och positionen tas där platshållaren hamnar.

**Lucka.** Mutanten `prompt.rfind(content)`, som söker i hela prompten i stället för i `pre`, överlever.
Inget test har ett tidigare meddelande vars text återkommer i ett senare. Testet med upprepad mening
(`test_I2_duplicate_sentence_located_in_order`) upprepar texten inom ett och samma meddelande.

Punkt 8 är **genomförd för det fall granskningen pekade ut**. Anmärkningen ovan återstår.

## Övriga punkter

| # | Kontroll | Bedömning |
| --- | --- | --- |
| 2 | Inte rättat. Läckaget ska redovisas i tolkningen och tas upp i planen för sprint 2. | Godtagbart för sprint 1. Det måste lösas före kärntestet. |
| 3 | `A_mean` och `A_first` finns i `Measurement`, i baslinjens rader och sammanfattning (`*_points_to_target`) och i utskriften. A (summa) är primärt. | OK |
| 4 | `test_A_positive_control_copy`: 40 uppgifter, kräver ≥ 36. Hoppas över på testmodellen. | Genomförd men **inte körd**. Gränsen 36/40 är inte motiverad. Svaret är målvärdet, och med det lexikala läckaget är testet en svag kontroll. Kör det med `NEXIS_REAL_MODEL` innan det räknas som godkänt. |
| 5 | `repetition_penalty=1.0` i `generate` (r. 161). | OK. Mutanten som tar bort det överlever, eftersom testmodellens generation_config redan har 1,0. |
| 6 | `when_A_not_target`: n, R = argmax A, R = mål. Utskrift när n > 0. | OK. Mutanten med omvänt villkor överlever, eftersom inget test kontrollerar värdena. |
| 7 | `transformers>=4.56`. | OK |
| 9 | `MAX_TOKENS = 1500`, ValueError visas i chatten (HTTP 500 med text). | OK. Vid gränsen blir attention-tensorerna 28 · 12 · 1500² · 4 B ≈ 3,0 GB i fp32, utöver ca 6,2 GB vikter. Det räcker med 14 GB-gränsen i `load_model`, men det är snävt. Kontrollen görs först efter genereringen. Inget test (mutanten överlever). |
| 10 | `answer_correct` matchar på ordgräns, med siffra som alternativ för talord. | Rätt. Mutanten som jämför delsträngar överlever: `test_parse_report_and_answer` har bara ett positivt fall. Lägg till "Max" mot "maximum". |
| 11 | Upprepad mening, token som korsar en meningsgräns, oberoende `mass_in_segments`, token-koppling på alla 8 `ITEMS`. | OK. Alla fyra mutanter som överlevde förra gången dödas nu: `mass_in_segments` = 1,0, sista överlappande segment, `locate_given` utan markör och `locate_given` med `rfind`. |
| 12 | I3-mutanten ändrar första, mittersta och sista parametern. `model_info`: revision, sha256 för chattmallen, versioner, dtype. | OK. Både kontrollsumma bara över första och bara över sista tensorn dödas. `_commit_hash` kan bli None om modellen laddas lokalt. |
| 13 | Kommentar i `load_model` om att "auto" här inte betyder HF:s "auto". | OK |
| 14 | `segment.py` r. 4 hänvisar till rätt testfil, och r. 86–89 nämner inte längre (0, 0). | OK |
| 15 | `mean_se` ger `None` vid n = 0. | OK, men utan test. |

## Nya fel eller inkonsekvenser

- **Anmärkning 1** ovan (`locate_segments` placerar korta meddelanden i mallens markör).
- `README.md` r. 15 säger fortfarande "alla svarstokens". Definitionen är nu "raderna som väljer svarets tokens",
  och README nämner inte `A_mean`/`A_first`. Texten är oprecis, men inget räknas fel.
- Inga andra fel hittade. Normeringen, GQA-vägen, `A_mean` (`clamp(min=1)` skiljer sig från bruteforce bara
  för tomma segment, och sådana kan inte uppstå), baslinjens `A_report` med uttrycklig systemprompt, och
  `chat.py`:s förskjutning av meddelandeindex vid systemprompt är korrekta.

## Sammanfattning

Dom: **godkänd med anmärkningar.**

1. `attention.py` r. 125: `pre.rfind` placerar meddelanden som "e", "m" och "end" i `<|im_end|>`. Det ger ett
   tyst fel i chatten. Sök i meddelandets eget fönster.
2. Testlucka: ett meddelande vars text återkommer i ett senare meddelande saknas (mutanten
   `prompt.rfind` överlever).
3. Den positiva kontrollen (punkt 4) är inte körd. Kör den på den riktiga modellen. Gränsen 36/40 och den
   svaga diskrimineringen bör redovisas.
4. Det saknas test som kan fallera för `answer_correct` (negativa fall), `when_A_not_target`, `MAX_TOKENS` och
   `repetition_penalty`.
5. README:s A-definition behöver uppdateras till de rättade raderna och de två jämförelsemåtten.
6. Punkt 2 (lexikalt läckage) är fortfarande öppen och måste lösas före kärntestet.
