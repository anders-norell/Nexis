# Grind 3 (sprint 3b) – rapport, 26 nov 2026

Plan: projektet `claude/nexis_v3_sprint3b_plan.md`. Förregistrering: `sprint3b_prereg.md`. Låsning: `sprint3b_lock.json`.
Testresultat (en körning): `sprint3b_test.json`. Efterhand: `sprint3b_posthoc.json`. Granskning: `sprint3b_review.md`.

## Utfall mot låsta kriterier (k = 4, 300 nya testtexter, testämnen)
- K1 kommutering: ε_G 0,142 mot identitet 0,190, z 26,5 – **klarat**.
- K2 nytta: Y-förlust S 0,320 mot karikatyren 0,559, z 7,0 – **klarat**.
- K3 trohet: ρ 0,627 mot slumpprojektion 0,392, 0,21 ± 0,04 – **klarat**.
- Formellt: grind 3 klarad. Procedur och låsning kontrollerade av granskaren, inget läckage av facit.

## Vad det betyder i sak (granskningen)
- K1 är i praktiken sant per konstruktion: en konstant drift slår G (ε 0,132), drift per tokenposition slår G klart, och en
  otränad P klarar K1. S kodar tokenposition starkt (R² 0,88). Ingen dynamik utöver "var i svaret jag är" är visad.
- K3:s nollmodell är för svag: PCA till 4 dimensioner bevarar avstånden bättre (0,66 mot 0,60), tokenposition ensam 0,56.
- K2 är verklig men retrospektiv: vid första token (före svaret) ingen vinst (0,516 mot konstant 0,500); vid andra raden
  stor vinst (0,287 mot 0,446). S läser ur vilken mening det just skrivna ordet kopierades (felen kommer till 72–87 % ur
  lockmeningen). Mot konstant prediktion (efterhand, ej förregistrerat) z 5,6. Karikatyren var sämre än konstant, så
  den förregistrerade jämförelsen var svag.
- Öppen och intressant: S skiljer kopiering ur målmeningen från kopiering ur lockmeningen, fast P inte vet vilken mening
  som är vilken.

## Tolkning
Schemat är en självavläsning i efterhand ("jag hämtade svaret ur lockmeningen"), inte en självmodell som förutser utfallet
eller bär uppmärksamhetens dynamik. Lärdomar till nästa förregistrering: K1 mot drift per position, K2 mot konstant och
helst bara med information från före svaret, K3 mot PCA med samma k.

## Anmärkningar
- Klockslagen i förregistreringen ("ca 07:10", "ca 07:30") stämmer inte med arbetsytans klocka; ordningen styrks av
  commit-historiken (förregistrering 2b4…, lås 29bc351, test 07b9ce8). Förregistreringen är låst och ändras inte.
- Låsfilen hashar inte datafilerna.
