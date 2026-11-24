# Granskning av Nexis v3, sprint 1

Granskare: fristående Claude-instans utan bygghistorik, 24 nov 2026.
Underlag: `v3/` som den låg vid granskningen. Inga befintliga filer har ändrats.
Testsviten: `python3 -m pytest -q` ger 38 passed (4,2 s, testmodellen). Miljö: torch 2.15, transformers 5.24.

Vad jag själv har kört (allt under `/tmp`):
- **Mutationsprov:** 18 mutanter av produktionskoden (`/tmp/mut/run.py`). Var och en kördes mot hela testsviten.
- **Qwen-lik tokenizer:** Qwen2:s pre-tokenizer-regex och ByteLevel, plus Qwen2.5:s chattmall med automatisk systemprompt
  (återgiven ur minnet). Jag använde den för att pröva offset_mapping och segmentkopplingen på genererade uppgifter
  (`/tmp/rev/qwenlike.py`). Den riktiga Qwen-tokenizern gick inte att ladda ner (huggingface.co blockeras, 403).
- **Läckage- och balansanalys av textgeneratorn** (`/tmp/rev/leak.py`).

## Dom

**Godkänd med anmärkningar.** Koden är ren och väl testad mot sin egen definition, och invarianterna I1–I3 håller.
Men definitionen av A har ett tidsfel: den mäter inte attention vid det steg där svarets första token väljs.
Det bör rättas innan A blir grund för kärntestet. Textgeneratorn läcker dessutom målet lexikalt till 100 %.
Det är inget fel, men det bestämmer hur baslinjen får tolkas.

---

## Invarianterna

### I1: A summerar till 1 och ändras när texten ändras

**(a) Implementation.** Korrekt mot den definition som står i docstringen (`attention.py` r. 3–10).
A = raw / raw.sum() gör att summan blir 1 per konstruktion. Kontrollen `check_sums_to_one` kan därför bara
slå larm vid NaN (raw.sum() = 0) eller negativa värden. Det är bra att den finns, men den testar nästan ingenting.

**(b) Fallerar testet om invarianten bryts?** Ja, i praktiken genom `test_I1_matches_bruteforce`, inte genom
mutanttestet. `test_I1_mutants_fails` kör bara kontrollfunktionerna på påhittade listor (A·1,3 och [0,25]·4).
Det visar att kontrollfunktionerna kan säga nej, men inte att ett fel i `attention_per_segment` skulle upptäckas.
Mina mutanter:

| Mutant | Utfall |
| --- | --- |
| frågepositioner förskjutna ett steg (`q_offbyone`) | dödad (bruteforce) |
| fel lager (`l+1`) | dödad (bruteforce) |
| bara sista query-token | dödad (bruteforce) |
| bara huvud 0 | dödad (bruteforce) |
| medel per token i stället för summa | dödad (bruteforce) |
| `mass_in_segments` alltid 1,0 | **överlever** |

Bruteforce-testet fångar alltså implementationsfel. Det är en omskrivning av *samma* definition, så det kan
aldrig fånga ett fel i själva definitionen. Det gäller anmärkning 1 nedan: mutanten som rättar tidsfelet dödas
av bruteforce-testet. Testerna låser därmed fast den nuvarande definitionen.

**(c) Luckor.**
- Testet "ändras när texten ändras" körs på en slumpvis initierad modell. Det visar bara att A beror på indata,
  vilket gäller nästan vad som helst. Det finns ingen **positiv kontroll** av att A pekar dit den borde. Ett
  exempel på en sådan kontroll: på den riktiga modellen kopierar svaret ordagrant en ovanlig sträng ur mening k,
  och argmax A ska då vara k i nästan alla fall. Utan en sådan kontroll vet vi inte att A mäter något
  meningsfullt innan kärntestet bygger på den.
- `mass_in_segments` testas bara med 0 < x ≤ 1. Mutanten som sätter den till 1,0 överlever.
- Ingen test av bf16-vägen (`--dtype bfloat16`). Attention-vikterna avrundas då till bf16 innan `.float()`
  (`attention.py` r. 155).

### I2: Segmentgränserna stämmer med meningarna

**(a) Implementation.** `split_sentences` klarar de kända fallen: förkortningar, decimaltal, listnumrering,
citat och radbrytningar. `token_segment_ids` kopplar varje token till det första segment den överlappar.
Med Qwen-lik pre-tokenisering på 200 genererade uppgifter överlappar **ingen token mer än ett segment**.
Mönstret blev: ` My` … ` Daisy` `.\n` hör till mening 1, och `1` `.` (numreringen) hör inte till någon mening.
Skiljetecken och radbrytning slås ihop till en token (`.\n`, `.\n\n`), och den räknas till meningen. Det är
rimligt, men se anmärkning 6. Specialtokens får i en snabb tokenizer sina riktiga teckenintervall, inte (0, 0)
som docstringen säger (`segment.py` r. 89). De överlappar inget segment och får ändå −1, så resultatet blir rätt.

**(b) Fallerar testet?**

| Mutant | Utfall |
| --- | --- |
| kräver att token börjar inom segmentet (tappar ledande blanksteg) | dödad (`test_I2_tokens_map_to_sentences`) |
| tappar sista tecknet i segmentet | dödad |
| förkortningar avstängda | dödad |
| ingen delning vid radbrytning | dödad |
| blanksteg tas med i segmentet | dödad |
| token till *sista* överlappande segment i stället för första | **överlever** |
| `locate_given` utan markör (söker från början varje gång) | **överlever** |
| `locate_given` med `rfind` (sista förekomsten) | **överlever** |

De tre överlevande visar att inget test har tokens som korsar en meningsgräns eller texter som förekommer två gånger.
I baslinjen är det i dag ofarligt, eftersom meningarna är unika och inga gränstokens uppstår. Det blir farligt
i kärntestet om samma mening upprepas i samtalet, eller om schemat senare skrivs in som text i prompten.

**(c) Luckor.**
- `check_boundaries(prompt, locate_given(...))` i `test_I2_given_segments_located` är nästan tautologisk.
  `locate_given` bygger segmenten med `find`, så `source[start:end] == text` gäller alltid.
- Token-kopplingen (`decode(toks).strip() == s.text`) testas bara på chattexemplet, inte på de genererade
  uppgifterna med numrering. Den bör köras på alla `ITEMS` och med den riktiga tokenizern.
- `locate_segments` (`attention.py` r. 92) söker från teckenposition 0. Qwen:s automatiska systemprompt
  ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") finns inte i `messages`. Om en
  användares text i chatten råkar vara en delsträng av den (t.ex. "You are", "helpful") placeras segmentet i
  systemprompten. Det berör bara chatten, men det är ett tyst fel.
- `segment.py` r. 4 hänvisar till `tests/test_segment.py`, som inte finns.

### I3: Basmodellens vikter är orörda

**(a) Implementation.** Korrekt. sha256 hashas över `state_dict` i namnordning med dtype och råa bytes, och det
fungerar även för bf16 via `.view(torch.uint8)`. `freeze` sätter eval-läge och `requires_grad=False`.

**(b) Fallerar testet?** Ja. Mutanter som hashar bara första tensorn, bara sista eller hoppar över embedding
dödas, och det gör även mutanten utan `freeze`. Mutanttestet ändrar dock bara *första* parametern
(`embed_tokens`). En kontrollsumma som bara hashade embedding skulle alltså överleva. Ändra hellre ett element
i en slumpvis vald eller sista parameter, eller i flera.

**(c) Luckor.**
- Kontrollsumman jämför bara före och efter *inom en körning*. Den visar inte att det är rätt modell som
  laddades. Spara modellens revision (`model.config._commit_hash`, eller sha256 för safetensors-filen) och
  chattmallens hash i resultatfilen, så att körningar på olika datorer kan jämföras.
- Förändringar genom hooks eller ändrad forward syns inte i `state_dict`. Det spelar ingen roll i sprint 1,
  men det blir relevant när LoRA och schema kommer. LoRA-moduler lägger till nya nycklar och ändrar därför
  summan, vilket är rätt beteende. Då behövs en kontrollsumma *över basvikterna specifikt*.
- På den riktiga modellen hoppas mutanttestet över (> 50 M parametrar). Det är rimligt.

---

## A-definitionen i `nexis3/attention.py`

1. **Frågepositionerna är förskjutna ett steg (allvarligast).** Raden `q = range(n_p, len(full))` (r. 146) tar
   raderna för svarstokens. I en kausal modell beräknar rad *t* sannolikheterna för token *t+1*. Svarets första
   token väljs alltså vid rad `n_p − 1` (sista prompt-token, "\n" efter `assistant`), och den raden är **inte med**.
   Rad `len(full) − 1` (sista svarstoken, som förutsäger EOS) är med. För korta svar som "Daisy." eller "Rex"
   (1–3 tokens) innebär det att just hämtningssteget, när värdet plockas ur målmeningen, tas bort ur måttet,
   medan steget som väljer punkt eller EOS räknas. Fallbacken när svaret saknas tokens använder `[n_p − 1]`.
   Den är alltså rätt, och den är inkonsekvent med huvudfallet.
   **Åtgärd:** `q = range(n_p − 1, len(full) − 1)`, eller inkludera båda ändarna och dokumentera valet.
   Uppdatera bruteforce-testet i samma veva. Redovisa gärna A för det första svarssteget separat, eftersom det
   är det renaste måttet på var hämtningen skedde.
2. **A är summa över tokens, inte medel.** Långa meningar får mer massa. Meningsmallarna varierar från 30 tecken
   (dog) till 56 (weather). Medel per token gav i mutationsprovet en skillnad på 0,04 i A redan på testmodellen.
   Det påverkar argmax A direkt. **Åtgärd:** redovisa båda varianterna, eller jämna ut meningslängden i tokens i
   generatorn. Valet bör vara låst före kärntestet.
3. **Lager:** `out.attentions[l]` är lager *l* (ingen embedding-post). `default_layers(28)` = 9–17, i enlighet
   med README. Korrekt.
4. **GQA:** `output_attentions` ger vikter per query-huvud (12) efter `repeat_kv`. Medel över 12 huvuden ger
   varje KV-grupp lika vikt. Korrekt, och testmodellen (4 huvuden, 2 KV) prövar vägen.
5. **Mallmarkörer och systemprompt** räknas inte in i A. De ingår i nämnaren för `mass_in_segments`. Det är
   korrekt, men `mass_in_segments` blir sannolikt liten på riktiga modellen, eftersom första token fungerar som
   attention-sänka. A är då fördelningen av en liten rest. Redovisa den och följ den.
6. **Numreringen ("1.", "2.") räknas inte till meningen**, men skiljetecken och radbrytning (`.\n`) gör det.
   För rapportfrågan ("Reply with the number only") är numreringstokens troligen det modellen tittar på när den
   skriver "3". Därför är `A_report` över meningarna ett svagt mått på vad rapporten bygger på. Tolka det inte
   som "attention under rapporten".
7. **Systemprompten:** Qwen2.5:s mall lägger tyst till en standardsystemprompt när ingen anges. Testmallen i
   `conftest.py` gör inte det, så testerna prövar aldrig den vägen. **Åtgärd:** ange en uttrycklig systemprompt
   i baslinjen och i kärntestet. Det gör körningen oberoende av mallversion, och schemafasen kan ändra
   systemprompten kontrollerat.
8. **Svar som ges som text** (`answer=...`, r. 185) tokeniseras fristående. Det kan skilja sig från hur samma
   text tokeniseras i sitt sammanhang. Det är ofarligt i dag, men i kärntestet bör tokens från `generate`
   återanvändas när det går.

---

## Textgeneratorn (`nexis3/textgen.py`)

- **Entydighet: god.** Svarsvärdet förekommer bara i målmeningen och aldrig i frågan, och det kontrolleras.
  Krocken "Friday" (bageri och schack) fångas av omdragningen.
- **Balans:** målpositionen är exakt balanserad när n är delbart med 4, i alla frön jag provade (0–2).
  Målets ämne är inte balanserat: frö 0, n = 100 ger music 15 och chess 4. Eftersom ämnet bestämmer
  meningslängden (punkt 2 ovan) blandas längd ihop med mål.
- **Läckage, viktigt för tolkningen.** En enkel ordöverlappning mellan frågan och meningarna (innehållsord)
  pekar ut målet i **100 av 100** uppgifter, utan oavgjorda fall. Ingen icke-målmening delar ett enda
  innehållsord med frågan. Exempel: "dog/neighbour's/called", "forecast/mountains/tomorrow", "Erik", "Lisa".
  Följder:
  - A på målet kan förklaras av ytlig matchning (induktionshuvuden och kopiering). Det betyder inte att A mäter
    "fokus" i någon djupare mening.
  - Rapporten R = mål kan modellen räkna ut ur innehållet, utan tillgång till sin attention. R = mål är därför
    väntat även för ett system som saknar självinsyn.
  - Det spelar ingen roll om meningen bara ska vara rätt. För kärntestet (följer R schemat eller faktisk
    attention?) är det dock nödvändigt med uppgifter där faktisk attention och "uppenbart svar" skiljer sig åt.
    Annars pekar alla tre mot samma mening. Planera t.ex. för frågor med parafras utan ordöverlappning, eller
    distraktormeningar som delar ord med frågan.
- **`answer_correct` (r. 99–102)** jämför delsträngar: "Max" godkänns i "maximum", "red" i "ordered", "oak" i
  "cloak", och "the answer is not red" räknas som rätt. **Åtgärd:** matcha på ordgräns (`\b…\b`), samma regex
  som generatorn använder.
- **`parse_report` (r. 88–93)** tar första siffran 1–4. "Sentence 10" tolkas som 1, och "the second sentence"
  blir None. Det är godtagbart, men räkna `R_unparsable` och spara råtexten. Båda görs redan.

---

## Baslinjeskriptet (`baseline.py`)

- **Wilson-intervallet är korrekt.** Kontrollerat: 50/100 ger 0,404–0,596, och 0/25 ger 0–0,133.
- **Binomialt SE är korrekt**, men det blir 0 vid p = 0 eller 1. Använd Wilson när du redovisar, vilket görs.
  `mean_se` kraschar vid n = 0 (r. 47). Det anropas bara på alla rader, så det är ofarligt.
- **`R_equals_A_argmax` (r. 85)** är svår att tolka när både R och argmax A oftast pekar på målet. Chansnivån
  är då inte 0,25. Lägg till de betingade måtten **R = argmax A | argmax A ≠ mål** och **R = mål | argmax A ≠ mål**.
  Det är exakt de fall som kärntestet handlar om, och baslinjen bör visa hur många de är och hur R beter sig där.
- `"chance": 0.25` är chans för en jämn gissning. För argmax A är den relevanta nollan fördelningen över
  positioner (`A_argmax_position_counts`), eftersom närhet till slutet och längd snedvrider. Det är bra att
  den sparas.
- `A_report` sparas men sammanfattas inte. Det är rimligt med tanke på punkt 6 ovan.
- Spara modellens revision, chattmallens hash samt versioner av torch och transformers i resultat-JSON (jfr I3).

---

## Riktiga Qwen-modellen på Windows CPU

- **Minne:** fp32 kräver ca 6,2 GB för vikterna, och `"auto"` ger fp32 (r. 31–32). Det är bra för precisionen,
  men "auto" betyder något annat i HF, där det är konfigurationens dtype (bf16). Byt namn eller dokumentera.
  `output_attentions=True` returnerar alla 28 lager: 28 · 12 · L² · 4 byte. Det är ca 0,1 GB vid L = 250
  (baslinjen), men **ca 5 GB vid L = 2 000**, och ett längre chattsamtal kan alltså ta slut på minnet.
  **Åtgärd:** samla bara de valda lagren med forward-hooks, eller begränsa samtalets längd i `chat.py`.
- **`dtype=`-argumentet** (r. 35) är den nya stavningen. Requirements tillåter transformers ≥ 4.46, och i
  äldre versioner heter argumentet `torch_dtype`. Där kan `dtype` ignoreras tyst, så att `--dtype bfloat16`
  ger fp32. **Åtgärd:** kräv transformers ≥ 4.56 (eller den version som faktiskt testats, 5.x) i
  `requirements.txt`, eller använd `torch_dtype`.
- **Girig avkodning är inte helt girig.** `generate` nollställer temperature, top_p och top_k, men inte
  `repetition_penalty`. Enligt min kännedom har Qwen2.5-Instruct:s `generation_config` ett värde över 1
  (ca 1,05–1,1; kontrollera). Det straffar tokens som redan finns i kontexten, *inklusive svarsvärdet i
  målmeningen*. Avkodningen blir ändå deterministisk, men den kopierar systematiskt mindre ur målet.
  **Åtgärd:** sätt `repetition_penalty=1.0` uttryckligen i `generate` (r. 125–129).
- **offset_mapping:** Qwen2TokenizerFast stöder det. Med Qwen:s pre-tokenizer-regex blir offseten exakta per
  token (blanksteg först i token, `.\n` sammanslagen), och tecken med flera byte (å, ä, ö) delar teckenintervall.
  Kopplingen fungerar. Bekräfta med `NEXIS_REAL_MODEL` att `test_I2_tokens_map_to_sentences` passerar.
- **Chattmall:** se punkt 7 under A-definitionen (automatisk systemprompt, inte testad).
- **Kodning:** `PYTHONUTF8=1` i ps1, och `write_text(..., encoding="utf-8")`. OK.
- **Tid:** sha256 över ca 6 GB tar storleksordningen 10–30 s per anrop. Det är OK i baslinjen (två gånger),
  men `/api/checksum` i chatten låser modellen under tiden.

---

## Anmärkningar, sorterade efter allvar

1. **`nexis3/attention.py` r. 146 – frågepositionerna är förskjutna ett steg.** Hämtningssteget för första
   svarstoken räknas inte, och steget för EOS räknas. Ändra till `range(n_p − 1, len(full) − 1)` och uppdatera
   bruteforce-testet. Redovisa gärna A för första svarssteget separat. Detta måste rättas före kärntestet.
2. **`nexis3/textgen.py` (TOPICS) – lexikalt läckage 100 %.** Frågan delar innehållsord bara med målmeningen.
   Baslinjens "A pekar på målet" och "R = mål" kan förklaras ytligt. Skriv det i tolkningen, och planera
   uppgifter där faktisk attention och uppenbart svar kan skilja sig, annars blir kärntestet odiskriminerande.
3. **`nexis3/attention.py` r. 157–160 – A summerar över tokens.** Det ger längdbias, och meningslängden hänger
   ihop med ämnet, som inte är balanserat som mål. Redovisa medel per token som jämförelse, eller jämna ut
   meningarnas tokenlängd. Lås valet före kärntestet.
4. **Positiv kontroll av A saknas** (`tests/test_invariants.py`). Lägg till ett test på den riktiga modellen där
   svaret kopierar en unik sträng ur mening k, och kräv argmax A = k i nästan alla fall.
5. **`nexis3/attention.py` r. 125–129 – `repetition_penalty` från Qwen:s generation_config används.** Sätt
   `repetition_penalty=1.0`.
6. **`baseline.py` r. 85 – betingade mått saknas.** Lägg till R = argmax A och R = mål, betingat på
   argmax A ≠ mål, med n.
7. **`requirements.txt` r. 2 – `transformers>=4.46` matchar inte `dtype=`** i `attention.py` r. 35. Höj
   kravet eller använd `torch_dtype`.
8. **Chattmallens automatiska systemprompt** (Qwen) testas inte, eftersom `conftest.py` r. 19–22 saknar den.
   Ange en uttrycklig systemprompt i `baseline.py` r. 57 (`msgs`). Lägg in en standardsystemprompt i testmallen,
   så att `locate_segments` prövas mot den. Rätta `locate_segments` (`attention.py` r. 92) så att sökningen
   börjar efter systemblocket.
9. **`chat.py` / `attention.py` r. 144 – `output_attentions` för alla 28 lager** sprängs vid långa samtal.
   Använd hooks för bara de valda lagren, eller begränsa längden.
10. **`nexis3/textgen.py` r. 102 – `answer_correct` jämför delsträngar.** Matcha på ordgräns.
11. **Testluckor från mutationsprovet:** lägg till test för (a) en mening som förekommer två gånger
    (`locate_given` utan markör och med `rfind` överlever), (b) en token som korsar en meningsgräns
    (sista-överlapp-mutanten överlever), och (c) ett oberoende värde för `mass_in_segments` i bruteforce-testet
    (mutanten med 1,0 överlever). Kör token-koppling (`decode == text`) på alla genererade uppgifter.
12. **I3-mutanten (`tests/test_invariants.py` r. 186) ändrar bara första parametern.** Ändra även sista eller
    en slumpvis vald. Spara dessutom modellens revision och chattmallens hash i `baseline.py`-resultatet.
13. **`dtype="auto"` → fp32** (`attention.py` r. 31) avviker från HF:s betydelse. Dokumentera eller byt namn.
14. **Docstrings:** `segment.py` r. 4 (testfilen finns inte) och r. 89 (specialtokens har inte (0, 0) i snabba
    tokenizers). Rätta texten.
15. **`baseline.py` r. 47 – `mean_se` vid n = 0** ger division med noll. Det är ofarligt i dag.
