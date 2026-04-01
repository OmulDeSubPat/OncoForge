# Legenda grafice

Acest fisier explica pe scurt ce reprezinta graficele exportate in folderul `grafice 30-3-2026`.
Textele au fost localizate in romana pentru prezentare si pentru citire mai usoara.

## pIC50
Reprezinta o masura logaritmica a potentei estimate.
In general, valori mai mari indica o activitate estimata mai buna impotriva tintei EGFR.

## QED
Este un scor de asemanare cu proprietatile tipice ale medicamentelor.
Valori mai mari sunt, de regula, mai favorabile, dar nu garanteaza singure utilitatea moleculei.

## RMSE
Este eroarea medie patratica exprimata pe aceeasi scara cu variabila prezisa.
Valori mai mici sunt mai bune pentru ca indica abateri mai mici intre predictii si valori reale.

## MAE
Este eroarea medie absoluta.
Valori mai mici inseamna predictii mai apropiate de datele reale.

## R2
Este coeficientul de determinare.
Valori mai apropiate de 1 indica faptul ca modelul explica mai bine variatia datelor.

## Incertitudine predictiva
Arata cat de nesigur este modelul in privinta unei predictii.
Valori mai mari inseamna ca estimarea trebuie interpretata cu mai multa prudenta.

## Scor de fezabilitate
Rezuma cat de realist pare un candidat din punct de vedere chimic si practic.
Valori mai mari sunt, in general, mai bune.

## Scor de pregatire experimentala
Indica daca un candidat este mai aproape de a merita testare ulterioara.
Valori mai mari sugereaza prioritate mai buna pentru validare.

## Scor de consens intre baze de date
Arata cat de bine este sustinut un candidat de surse publice independente.
Valori mai mari indica dovezi externe mai consistente.

## Scor de noutate
Masoara cat de diferit este candidatul fata de moleculele deja cunoscute.
Valori mari pot fi utile, dar trebuie echilibrate cu fezabilitatea si dovezile existente.

## Scor de aplicabilitate
Arata cat de aproape este un candidat de zona de date in care modelul invata bine.
Valori mici inseamna risc mai mare ca predictia sa fie nesigura.

## Risc de reward hacking
Sugereaza cat de probabil este ca o molecula sa para buna numeric fara sa fie convingatoare chimic.
Valori mai mici sunt preferabile.

## Curbe de antrenare RL
Arata cum evolueaza recompensa obtinuta de agent pe parcursul episoadelor.
O tendinta ascendenta poate fi favorabila, dar trebuie interpretata impreuna cu graficele de audit, fezabilitate si dovezi externe.

## Heatmap de suport multi-agent
Fiecare coloana este un candidat, iar fiecare linie este o sursa de suport.
Valorile mai mari inseamna un acord mai puternic intre agentii sau filtrele specializate.

## Studii de ablatie
Aceste grafice compara variante ale aceluiasi pipeline in care o componenta este eliminata sau redusa.
Daca performanta scade clar dupa eliminare, componenta respectiva are o contributie relevanta in selectie.

## Provenienta temporala
Graficele sau tabelele temporale trebuie citite impreuna cu sursa anului asociat fiecarei molecule.
Anul poate proveni direct din datele brute sau din documentul ChEMBL asociat si trebuie interpretat cu prudenta cand acoperirea nu este completa.
