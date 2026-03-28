# README

De bestanden in deze folder zijn de automatische fietstellingen (EcoCounter) van AWV.

Het bestand `sites.csv` bevat de informatie van de sites of plaatsen waar tellingen plaatsvinden.
De bestanden `data-<Jaar>-<Maand>.csv` bevatten de fietstellingen zelf.

Bron van de fietstelgegevens “Eco Display Classic Budstraat, ID 143, site-id 100038932” is de stad Kortrijk.

De bestandstructuur is de volgende.

## `sites.csv`
 
|  Kolom    | type       | beschrijving              |
|---        |---         |---                        |
| site ID   | int        | ID (primary key)          |
| site nr   | int        | Ecocounter ID van site    |
| long      | float      | WGS84 longitude           |
| lat       | float      | WGS84 Latitude            |
| naam      | text       | naam site                 |
| domein    | text       | domein naam van site      |
| wegnr     | text       | Wegnummer van site        |
| district  | text       | Nummer wegendistrict      |
| gemeente  | text       | Gemeente van site         |
| interval  | int        | meetinterval in minuten   |
| datum_van | datum      | datum van installatie     |

## `richtingen.csv`
|  Kolom    | type       | beschrijving              |
|---        |---         |---                        |
| site ID   | int        | ID (primary key)          |
| richting  | Tekst      | IN, OUT of IN/OUT         |
| naam      | text       | Omschrijving richting     |


## `data-<jaar>-<maand>.csv`

|  Kolom    | type       | beschrijving              |
|---        |---         |---                        |
| site ID   | int        | FKey naar Site            |
| richting  | Tekst      | IN, OUT of IN/OUT         |
| type      | Tekst      | Type telling              |
| van       | datum/tijd | start van interval        |
| tot       | datum/tijd | einde van interval        |
| aantal    | int        | aantal geteld in interval |

Type telling kan zijn: VOETGANGERS, FIETSERS, PAARDEN, AUTOS, BUSSEN, 
MINIBUSSEN, NIET GEDEFINIEERD, MOTORFIETSEN, KAYAKS
