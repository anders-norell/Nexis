# Nexis v3 – sprint 1

Styrdokument: claude.ai-projektet "AST implementation", `claude/nexis_v3_tradintro.md`.
Sprint 1: chatt med basmodellen och mätning av attention per mening. Inget schema, ingen LoRA.

## Basmodell

**Qwen/Qwen2.5-1.5B-Instruct** [Claudes förslag]. Den har Apache 2.0-licens och kräver ingen inloggning
(Llama 3.2 kräver godkänd licens och HF-konto). Den följer instruktioner bättre än 1B-alternativen,
har 28 lager och 12 attention-huvuden, och går på CPU (ca 6 GB RAM i float32, 3 GB i bfloat16).
Den laddas med `attn_implementation="eager"` så att attention-vikterna kan läsas ut.

## Mätningen (A)

För varje svar och varje mening s i kontexten tas medelvärdet över mittlagren (default lager 9–17
av 28), alla huvuden och alla svarstokens av den attention som går till meningens tokens. Sedan
normeras över meningarna så att summan blir 1. Den andel attention som går till allt annat
(mallmarkörer, systemprompt, första token) redovisas separat som `mass_in_segments`.
Valet av lager och normeringen är [Claudes förslag].

## Filer

| Fil | Vad |
| --- | --- |
| `nexis3/segment.py` | meningsindelning och token → mening |
| `nexis3/attention.py` | laddning (fryst modell), generering, A per mening, viktkontrollsumma |
| `nexis3/textgen.py` | testtexter: fyra meningar om olika ämnen + fråga om en av dem (frö, balanserad målposition) |
| `nexis3/invariants.py` | invarianterna som kontrollfunktioner |
| `chat.py`, `chat.html` | chatt i webbläsaren, A per mening bredvid svaret |
| `baseline.py` | baslinjen: A pekar ut målet? rapport R = målet? |
| `tests/` | invarianttester med mutanter som måste fallera; röktest |
| `setup_windows.ps1` | installation och körning på Windows |

## Invarianter (tests/test_invariants.py)

- **I1** A summerar till 1 över meningarna och ändras när texten ändras. Stämmer med en oberoende
  loopimplementation av definitionen.
- **I2** Meningsgränserna stämmer i kända exempel (förkortningar, decimaltal, listnumrering, svenska) och
  för genererade texter. Tokens kopplas till rätt mening.
- **I3** Basmodellens vikter är orörda: kontrollsumman (sha256) är densamma före och efter användning.
  Modellen är fryst och i eval-läge.

Varje invariant har ett mutanttest som kör kontrollen på en trasig variant och kräver larm.
Testerna körs på en liten slumpmodell (`pytest`). Med `NEXIS_REAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct`
körs de på den riktiga modellen.

## Körning

```
pip install -r requirements.txt
python -m pytest                      # testmodellen, några sekunder
python chat.py                        # http://localhost:7860
python baseline.py --n 100 --seed 0   # results/baseline_s0.json
```
På Windows: `powershell -ExecutionPolicy Bypass -File setup_windows.ps1 install|test|baseline|chat`.
