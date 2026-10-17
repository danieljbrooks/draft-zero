# Common cards' GIH win rates: the strongest player's self-play against 17Lands

*Card data retrieved from 17Lands: [FDN Sealed Card Data](https://www.17lands.com/card_data?expansion=FDN&format=Sealed&time_period=ALL_TIME), all time, as of 16 October 2026; the Premier Draft column is the repo's 17Lands reference (`assets/reference/FDN_gih.json`, from 17Lands' public FDN data). Data behind docs/034 §12.*

Ours: 6,922 self-play games of E1's versions 25-36 and E6 (from version 36), each seat once. The 87 commons with at
least 100 games in hand here and 300 in 17Lands Sealed. Each GIH WR is relative to its own side's mean over these
commons (ours 50.7%, 17Lands Sealed 54.7%, Premier Draft 54.2%); ± is a 95% interval. Sorted by 17Lands Sealed.

| Common | Games in hand, ours | GIH WR vs mean, ours | Games in hand, 17Lands Sealed | GIH WR vs mean, 17Lands Sealed | Ours − Sealed | 17Lands Premier Draft |
|---|---|---|---|---|---|---|
| Scoured Barrens | 687 | +3.1 (±3.7) | 1,879 | +5.2 (±2.2) | −2.2 | +0.7 |
| Luminous Rebuke | 1,427 | +6.5 (±2.6) | 5,399 | +5.2 (±1.3) | +1.2 | +3.5 |
| Dazzling Angel | 1,621 | +9.8 (±2.4) | 6,151 | +5.1 (±1.2) | +4.7 | +3.6 |
| Helpful Hunter | 1,746 | +7.1 (±2.3) | 5,885 | +5.0 (±1.3) | +2.1 | +3.3 |
| Banishing Light | 1,504 | +4.7 (±2.5) | 6,803 | +4.7 (±1.2) | −0.0 | +3.3 |
| Cathar Commando | 1,280 | +3.3 (±2.7) | 5,028 | +4.3 (±1.4) | −1.1 | +1.4 |
| Bake into a Pie | 1,964 | +5.9 (±2.2) | 5,824 | +4.2 (±1.3) | +1.7 | +3.8 |
| Felidar Savior | 1,629 | +10.1 (±2.4) | 5,601 | +4.1 (±1.3) | +6.0 | +3.2 |
| Burglar Rat | 2,085 | +1.3 (±2.1) | 5,009 | +4.0 (±1.4) | −2.7 | +2.1 |
| Healer's Hawk | 1,633 | +6.7 (±2.4) | 5,716 | +3.9 (±1.3) | +2.7 | +3.3 |
| Blossoming Sands | 360 | −0.4 (±5.2) | 1,723 | +3.9 (±2.3) | −4.3 | −0.4 |
| Make Your Move | 889 | +1.7 (±3.3) | 4,158 | +3.7 (±1.5) | −2.1 | +1.1 |
| Tranquil Cove | 711 | −0.3 (±3.7) | 1,440 | +3.5 (±2.5) | −3.8 | +1.5 |
| Burst Lightning | 1,781 | −1.0 (±2.3) | 4,539 | +3.3 (±1.4) | −4.3 | +3.7 |
| Eaten Alive | 1,979 | +3.8 (±2.2) | 5,343 | +3.1 (±1.3) | +0.7 | +2.0 |
| Pilfer | 904 | −3.5 (±3.3) | 2,078 | +2.9 (±2.1) | −6.4 | +1.5 |
| Bloodfell Caves | 579 | −5.4 (±4.1) | 1,133 | +2.9 (±2.9) | −8.3 | +0.2 |
| Squad Rallier | 814 | +6.4 (±3.4) | 2,196 | +2.5 (±2.1) | +3.8 | +1.4 |
| Wind-Scarred Crag | 434 | −2.7 (±4.7) | 1,156 | +2.5 (±2.9) | −5.2 | −0.5 |
| Vanguard Seraph | 835 | +3.6 (±3.4) | 3,510 | +2.4 (±1.6) | +1.2 | +0.3 |
| Gorehorn Raider | 1,002 | +0.3 (±3.1) | 2,151 | +2.3 (±2.1) | −1.9 | +2.5 |
| Stab | 2,252 | +2.9 (±2.1) | 5,607 | +2.2 (±1.3) | +0.7 | +3.7 |
| Uncharted Voyage | 1,724 | −3.6 (±2.4) | 2,351 | +2.2 (±2.0) | −5.7 | +2.1 |
| Prideful Parent | 1,321 | +6.3 (±2.7) | 4,950 | +2.1 (±1.4) | +4.2 | +0.6 |
| Bigfin Bouncer | 1,885 | +1.1 (±2.3) | 3,403 | +1.9 (±1.7) | −0.9 | +2.9 |
| Wary Thespian | 827 | +6.1 (±3.4) | 4,339 | +1.9 (±1.5) | +4.1 | +1.2 |
| Bite Down | 940 | +5.0 (±3.2) | 5,111 | +1.6 (±1.4) | +3.4 | +1.4 |
| Dismal Backwater | 893 | −3.3 (±3.3) | 1,500 | +1.6 (±2.5) | −4.9 | +2.1 |
| Sower of Chaos | 259 | +1.5 (±6.1) | 898 | +1.4 (±3.2) | +0.0 | +0.1 |
| Fleeting Flight | 1,120 | −1.1 (±2.9) | 3,173 | +1.2 (±1.7) | −2.3 | +2.3 |
| Evolving Wilds | 2,143 | −5.4 (±2.1) | 8,506 | +1.0 (±1.1) | −6.4 | +0.4 |
| Infestation Sage | 1,884 | +1.8 (±2.3) | 3,816 | +0.9 (±1.6) | +0.9 | +2.5 |
| Goblin Boarders | 642 | −1.9 (±3.9) | 2,144 | +0.8 (±2.1) | −2.7 | −0.5 |
| Inspiring Paladin | 822 | +7.0 (±3.4) | 4,068 | +0.8 (±1.5) | +6.2 | −0.3 |
| Swiftwater Cliffs | 531 | −7.5 (±4.2) | 806 | +0.8 (±3.4) | −8.3 | +0.0 |
| Soul-Shackled Zombie | 949 | +7.0 (±3.1) | 2,463 | +0.7 (±2.0) | +6.3 | +0.9 |
| Think Twice | 1,833 | −3.3 (±2.3) | 3,243 | +0.7 (±1.7) | −4.0 | +2.4 |
| Involuntary Employment | 578 | +0.4 (±4.1) | 909 | +0.7 (±3.2) | −0.3 | +1.4 |
| Fake Your Own Death | 1,087 | −9.0 (±2.9) | 2,501 | +0.5 (±1.9) | −9.5 | +1.6 |
| Witness Protection | 649 | −1.8 (±3.8) | 2,471 | +0.5 (±2.0) | −2.3 | −1.4 |
| Quick-Draw Katana | 152 | −3.9 (±7.9) | 587 | +0.5 (±4.0) | −4.5 | −3.3 |
| Beast-Kin Ranger | 883 | +7.2 (±3.3) | 4,197 | +0.5 (±1.5) | +6.7 | +1.1 |
| Llanowar Elves | 1,051 | +3.8 (±3.0) | 5,160 | +0.5 (±1.4) | +3.3 | +2.9 |
| Goldvein Pick | 1,027 | −5.7 (±3.0) | 4,442 | +0.4 (±1.5) | −6.0 | −0.6 |
| Strix Lookout | 1,424 | −0.0 (±2.6) | 3,218 | +0.1 (±1.7) | −0.1 | +0.8 |
| Armasaur Guide | 174 | +5.1 (±7.4) | 1,562 | −0.1 (±2.5) | +5.2 | −3.7 |
| Vampire Soulcaller | 974 | +1.3 (±3.1) | 3,198 | −0.2 (±1.7) | +1.5 | −0.2 |
| Gutless Plunderer | 840 | +0.7 (±3.4) | 2,770 | −0.2 (±1.9) | +0.9 | −0.3 |
| Treetop Snarespinner | 685 | +5.0 (±3.7) | 4,160 | −0.3 (±1.5) | +5.2 | +0.4 |
| Bushwhack | 892 | −1.9 (±3.3) | 4,782 | −0.4 (±1.4) | −1.5 | −0.4 |
| Marauding Blight-Priest | 479 | +4.7 (±4.5) | 2,358 | −0.4 (±2.0) | +5.1 | −1.5 |
| Hungry Ghoul | 1,665 | +2.1 (±2.4) | 4,088 | −0.6 (±1.5) | +2.7 | +1.4 |
| Ambush Wolf | 614 | −1.5 (±4.0) | 2,989 | −0.6 (±1.8) | −0.8 | +1.1 |
| Gnarlid Colony | 738 | −0.5 (±3.6) | 4,379 | −0.7 (±1.5) | +0.1 | −1.2 |
| Jungle Hollow | 402 | −5.1 (±4.9) | 1,567 | −0.7 (±2.5) | −4.4 | −1.4 |
| Refute | 1,951 | −5.2 (±2.2) | 2,426 | −0.9 (±2.0) | −4.3 | +3.4 |
| Dwynen's Elite | 576 | +3.9 (±4.1) | 3,068 | −0.9 (±1.8) | +4.8 | −1.1 |
| Elfsworn Giant | 446 | +0.0 (±4.6) | 3,271 | −1.2 (±1.7) | +1.2 | −1.6 |
| Rugged Highlands | 325 | −3.6 (±5.4) | 1,029 | −1.2 (±3.0) | −2.3 | −0.3 |
| Sanguine Syphoner | 553 | +5.4 (±4.1) | 2,469 | −1.3 (±2.0) | +6.7 | −0.9 |
| Goblin Surprise | 734 | −4.5 (±3.6) | 1,960 | −1.4 (±2.2) | −3.0 | −1.6 |
| Cackling Prowler | 515 | +1.8 (±4.3) | 3,209 | −1.5 (±1.7) | +3.3 | −0.1 |
| Fanatical Firebrand | 463 | −4.6 (±4.5) | 1,703 | −1.6 (±2.4) | −3.0 | −1.2 |
| Campus Guide | 674 | −4.2 (±3.8) | 3,179 | −1.8 (±1.7) | −2.4 | −2.8 |
| Axgard Cavalry | 561 | −0.6 (±4.1) | 1,900 | −1.9 (±2.2) | +1.4 | −2.0 |
| Macabre Waltz | 772 | −2.9 (±3.5) | 2,248 | −2.1 (±2.1) | −0.8 | −2.3 |
| Hare Apparent | 300 | +1.7 (±5.7) | 1,029 | −2.1 (±3.1) | +3.8 | −3.5 |
| Lightshell Duo | 1,040 | −4.7 (±3.0) | 1,444 | −2.2 (±2.6) | −2.5 | +1.5 |
| Apothecary Stomper | 311 | +4.0 (±5.5) | 2,497 | −2.3 (±2.0) | +6.3 | −2.6 |
| Incinerating Blast | 521 | +0.2 (±4.3) | 1,762 | −2.4 (±2.3) | +2.6 | −1.8 |
| Courageous Goblin | 739 | +2.0 (±3.6) | 1,925 | −2.4 (±2.2) | +4.4 | −0.3 |
| Giant Growth | 487 | −3.2 (±4.4) | 2,142 | −2.5 (±2.1) | −0.7 | +0.0 |
| Fleeting Distraction | 1,653 | −4.1 (±2.4) | 2,315 | −2.6 (±2.0) | −1.5 | +0.7 |
| Thornwood Falls | 442 | −5.4 (±4.6) | 1,468 | −2.8 (±2.6) | −2.6 | −0.4 |
| Run Away Together | 485 | −13.3 (±4.3) | 1,677 | −3.0 (±2.4) | −10.3 | −2.7 |
| Elementalist Adept | 846 | −2.9 (±3.4) | 1,797 | −3.1 (±2.3) | +0.2 | −1.6 |
| Aegis Turtle | 193 | −8.7 (±7.0) | 669 | −3.6 (±3.8) | −5.1 | −4.3 |
| Sure Strike | 395 | −8.1 (±4.9) | 1,128 | −4.0 (±2.9) | −4.1 | −1.8 |
| Icewind Elemental | 905 | +0.1 (±3.3) | 2,288 | −4.0 (±2.0) | +4.1 | −0.4 |
| Erudite Wizard | 549 | −2.4 (±4.2) | 1,582 | −4.0 (±2.5) | +1.6 | −3.1 |
| Crypt Feaster | 121 | +9.7 (±8.7) | 732 | −4.4 (±3.6) | +14.1 | −3.8 |
| Gleaming Barrier | 197 | −2.9 (±7.0) | 762 | −5.5 (±3.5) | +2.5 | −6.4 |
| Firebrand Archer | 387 | −5.9 (±5.0) | 1,477 | −6.1 (±2.5) | +0.1 | −4.5 |
| Thrill of Possibility | 514 | −4.3 (±4.3) | 1,152 | −6.3 (±2.9) | +2.0 | −2.7 |
| Mocking Sprite | 482 | −3.3 (±4.5) | 1,425 | −6.7 (±2.6) | +3.3 | −4.2 |
| Grow from the Ashes | 259 | −4.3 (±6.1) | 2,359 | −6.8 (±2.0) | +2.5 | −6.1 |
| Tolarian Terror | 1,068 | −3.6 (±3.0) | 1,883 | −7.5 (±2.3) | +3.9 | −2.0 |
