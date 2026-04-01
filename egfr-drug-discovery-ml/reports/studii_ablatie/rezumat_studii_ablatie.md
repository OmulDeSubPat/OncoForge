# Studii de ablatie OncoForge

Acest document sintetizeaza studii de ablatie interne pentru componentele-cheie ale pipeline-ului.
Rezultatele descriu comportamentul in silico al strategiilor alternative si nu inlocuiesc validarea experimentala.

## Ranking Guardrails
### Rezultatele modelului curent
- Strategia principala: `scor_protejat_final`
- Pentru top `25` candidati, potenta estimata medie este `9.309`, iar riscul mediu de reward hacking este `0.000`.
- Rata `ready` este `0.000`, iar rata `audit pass` este `1.000`.
### Comparatie cu baseline intern
- Cea mai sigura strategie din acest studiu este `scor_protejat_final`, cu risc mediu `0.000`.
- Strategia cea mai slaba pe acelasi criteriu este `proxy_naiv`, cu risc mediu `0.090`.
- Diferentele trebuie interpretate ca efecte de ranking intern, nu ca dovada experimentala.
### Comparatie cu studii similare
- Comparatia cu literatura ramane provizorie deoarece `comparatii_literatura.csv` nu contine inca valori externe complete pentru acest studiu.
- In aceasta versiune, studiul de ablatie compara in primul rand variante interne ale pipeline-ului, conform artefactelor disponibile.

## Readiness Components
### Rezultatele modelului curent
- Strategia principala: `scor_readiness_complet`
- Pentru top `10` candidati, potenta estimata medie este `9.208`, iar riscul mediu de reward hacking este `0.000`.
- Rata `ready` este `0.900`, iar rata `audit pass` este `1.000`.
### Comparatie cu baseline intern
- Cea mai sigura strategie din acest studiu este `scor_readiness_complet`, cu risc mediu `0.000`.
- Strategia cea mai slaba pe acelasi criteriu este `fara_trasabilitate`, cu risc mediu `0.000`.
- Diferentele trebuie interpretate ca efecte de ranking intern, nu ca dovada experimentala.
### Comparatie cu studii similare
- Comparatia cu literatura ramane provizorie deoarece `comparatii_literatura.csv` nu contine inca valori externe complete pentru acest studiu.
- In aceasta versiune, studiul de ablatie compara in primul rand variante interne ale pipeline-ului, conform artefactelor disponibile.

## Generation Components
### Rezultatele modelului curent
- Strategia principala: `scor_generare_complet`
- Pentru top `10` candidati, potenta estimata medie este `9.247`, iar riscul mediu de reward hacking este `0.000`.
- Rata `ready` este `0.000`, iar rata `audit pass` este `1.000`.
### Comparatie cu baseline intern
- Cea mai sigura strategie din acest studiu este `scor_generare_complet`, cu risc mediu `0.000`.
- Strategia cea mai slaba pe acelasi criteriu este `fara_politica_generatorului`, cu risc mediu `0.000`.
- Diferentele trebuie interpretate ca efecte de ranking intern, nu ca dovada experimentala.
### Comparatie cu studii similare
- Comparatia cu literatura ramane provizorie deoarece `comparatii_literatura.csv` nu contine inca valori externe complete pentru acest studiu.
- In aceasta versiune, studiul de ablatie compara in primul rand variante interne ale pipeline-ului, conform artefactelor disponibile.

## Rl Components
### Rezultatele modelului curent
- Strategia principala: `prioritate_rl_completa`
- Pentru top `5` candidati, potenta estimata medie este `9.306`, iar riscul mediu de reward hacking este `0.000`.
- Rata `ready` este `0.800`, iar rata `audit pass` este `1.000`.
### Comparatie cu baseline intern
- Cea mai sigura strategie din acest studiu este `prioritate_rl_completa`, cu risc mediu `0.000`.
- Strategia cea mai slaba pe acelasi criteriu este `fara_prior_adaptiv`, cu risc mediu `0.000`.
- Diferentele trebuie interpretate ca efecte de ranking intern, nu ca dovada experimentala.
### Comparatie cu studii similare
- Comparatia cu literatura ramane provizorie deoarece `comparatii_literatura.csv` nu contine inca valori externe complete pentru acest studiu.
- In aceasta versiune, studiul de ablatie compara in primul rand variante interne ale pipeline-ului, conform artefactelor disponibile.
