# Error analysis (validation group 'thr', 46,818 train entities)

Decision rule: {'rule': 'threshold', 't': 0.75, 'r': 0.0, 'macro_f05': 0.9869516233341699}

## Summary

| entities | macro_f05 | macro_precision (entities with predictions) | macro_recall (entities with matches) | micro_precision | micro_recall | singletons | singleton_accuracy (predicted empty) | f05_non_singletons |
|---|---|---|---|---|---|---|---|---|
| 46818 | 0.987 | 0.9968 | 0.9664 | 0.9969 | 0.9656 | 2641 | 0.989 | 0.9868 |

## By country

| country | entities | macro_f05 | macro_precision (entities with predictions) | macro_recall (entities with matches) | micro_precision | micro_recall | singletons | singleton_accuracy (predicted empty) | f05_non_singletons |
|---|---|---|---|---|---|---|---|---|---|
| india | 18192 | 0.9874 | 0.997 | 0.9656 | 0.997 | 0.9642 | 1038 | 0.9904 | 0.9872 |
| us | 28626 | 0.9867 | 0.9966 | 0.9668 | 0.9968 | 0.9665 | 1603 | 0.9881 | 0.9866 |

## Where the true pairs go

| bucket | true pairs | share |
|---|---|---|
| found | 156933 | 96.56% |
| missed: low score | 3167 | 1.95% |
| missed: not blocked | 2008 | 1.24% |
| missed: one-to-one | 422 | 0.26% |

## False merges

| false merges | on singleton entities | predicted pairs |
|---|---|---|
| 493 | 31 | 157426 |

### Examples: false merges
| S1 name | S1 address | candidate name | candidate address | p |
|---|---|---|---|---|
| Piedmont Holdings | 1127 Cadillac Avenue, Dauphin Island, AL | Piedmont Holdings Holdings | 132 Cadillac Ave, Dauphin Island, Alabama | 0.94 |
| Pranika For Private Limited | Andheri West, Mumbai, 201, Maharashtra, Shanti Sadan, Near M | Fayevantage | #G-731 201, Shanti Sadan, Near Mc Donalds Lokhandwala Comple | 0.97 |
| Time Quest Pvt Ltd | 3Rd Flr, H.No.759 1125 A/B, 1064 Cresent Niwas, Parshi Pada, | Fluxkelopyra Sys | Flat #303, 3Rd Flr, H.No.759 1125 A/B, 1064 Cresent Niwas, P | 0.98 |
| American Program II | 35400 Gosling Lane, Orange County, VA | Rsne LP | 35400 GOSLING LN, ORANGE COUNTY, VA | 0.84 |
| Montgomery, Vickie N., Ph.D., P.C. Inc. | 1612 White Mountain Boulevard, Pinetop-lakeside, AZ | Montgomery, Víckie N., Ph.D., P.C. Inc | 1616 White Mountain Blvd, Pinetop-Lakeside, Arizona | 0.78 |
| Rajan Group | Third Floor, Electronic Sadan No. - I, Midc, Ttc Industrial  | LIMITED RAJAN GROUP | NO 83 THIRD FLOOR, MAHAPE, Maharashtra | 0.98 |
| Golden Property Associates | 1952 Oak Hill Drive, Town Of Pittsfield, WI | Noviavi | GEEN BAY, WI, 1952 OAK HILL DRIVE | 0.86 |
| SIB Shikshan Pvt Ltd | Tadoor, Karimnagar, Telangana, H.No 8-75 First Floor Near Ma | Vantagexylo | #8-75 FIRST FLOOR NEAR MARKANDIYA TEMPLE, THANGALLAPALLE, KA | 0.87 |
| Mobile Constructions | Mattathu Maruthimoottil Buildings Kumbanad P.O, Kumbanad, Pa | Mobile Services | #2-94 Mattathu Maruthimoottil Buildings Kumbanad P.o, Pathan | 0.96 |
| Ear Nose & Throat Care Inc. | Goshen, 1628 Aspen Drive, IN | Ear  Nose & Throat Care Inc | 163 Aspen Dr, Goshen, Indiana | 0.85 |
| Microtech Forex Corp | Office No. 1, Shed No. 2, Sector A, Huzur, Bhopal, Madhya Pr | Sri Microtech Corporation  Center |  | 0.76 |
| Bangalore North Finlease Private Limited | Bangalore, Karnataka, No 22, Sjngapura Village, Bangalore No | Limited Bangalore North Private Services | NO. 22, BANGALORE NORTH, BANGALORE, Karnataka | 0.94 |

### Examples: missed, not blocked
| S1 name | S1 address | candidate name | candidate address | p |
|---|---|---|---|---|
| Great Institute | 3701 Atlantic Street, Spokane, WA | Great Institute Trading |  | - |
| Behavioral Health Federal Partners Inc | 695 Muirfield Avenue, Simi Valley, CA | Behavioral Health Partners Inc Center |  | - |
| Pediatric Dentistry Allied Care LLC | 5 A Sheridan Village, Unit Apartment 2, Schenectady, NY | Pediatric Dentistry Asildn Care LLC | Schenectdy Township, 1 Sheridan Village, New York, # Apartme | - |
| Ghaziabad Fast Limited | Office No 130 Ist Floor, Cloud 9 Sec-1 Vaishali, Ghaziabad,  | Ghaziabad Fs Limited |  | - |
| Emerald Estate | T - 4, R - 44, Kishan Arcade, Zone - I M P Nagar, Bhopal, Ma | Eestate.Com | T - 4, Bhopal, MP | - |
| Good Foods Private Limited | Nandan, Plot No. 36/3, Nr. Bank Of India, Turbhe, Navi Mumba | Good Private Limited-Center |  | - |
| Sai Sparkling (India) Pvt Ltd | Flat No. S-1, Madhu Apartment, Plot No. 46, Suruchi Nagar, H | >> Sai (India) Pvt  Ltd Service | MP, Huzur, Flat No. S-#1 | - |
| Jenson, Mccray and Rochon Energy | 1556 Lake Road, Ontario, NY | jensonmccrayrochoncom | 1252 Lmke Rd, Ontario, New York | - |
| Christ Cathedral | 211 Hall Avenue, Hubbard, OH | Christ Cahtdeibrgl |  | - |
| Vashi (india) Private Limited | 104, Plot -92, Sector -19/C, Vashi, Navi Mumbai, Maharashtra | Ariaveo | Block D-84 104, Vashi, Navi Mumbai, MH | - |
| Inaaya Trading Private Limited | Plot No..-2/3, Gondarmau, Abbas Nagar Roa, Huzur, Bhopal, Ma | VERAARIASYN | BHOPAL, Madhya Pradesh, POT NO..-2, HUZUR | - |
| Inland Infotech (India) Private Limited | C/O Sandhya Sharma, F-18/5, Ews Madhuban Bapudh, Ghaziabad,  | lnland Infotech  (India) Private Limited | C/o Sandhya Sharma, Ghaziabad, UP | - |

### Examples: missed, low score
| S1 name | S1 address | candidate name | candidate address | p |
|---|---|---|---|---|
| Josey Hill Spruce LLC | 1008 Florida Street, Spokane, WA | Josey Hill Spruce  LLC | 1017- Florida Street, Spokane, Washington | 0.43 |
| Congdon Elite Ribbon | 1826 Mcgill Circle, Eufaula, OK | Congdon Elite |  | 0.07 |
| Zidynus Corp | 22288 Greenbush Road, Accomack County, VA | Zidynus |  | 0.17 |
| Ashoka & Sons Private Limited | 29, Mangal Marg Road, Bhopal, Madhya Pradesh | ASHOKA & SONS PRIVATE LIMITED |  | 0.33 |
| Womens Health Secure Care L.L.C. | 129 Jogee Road, Wawayanda, NY | Womens Health Secure Care |  | 0.38 |
| Shelton and Williams Roofing | 29720 Jeans Road, Unit Space 9, Veneta, OR | Shelton And Williams Roofing Inc. | 29721 Jeans Rd, # Space 9, Veneta, Oregon | 0.14 |
| BZC Foundation Private Limited | E-5/24, Arera Colony Ward No. 50, Arera Colony, Bhopal, Madh | BZC Foundation Pfirhate Limited |  | 0.29 |
| Ammons Preparatory Academy | Unit Suite 4, 6850 Main Street, NY, Amherst | Ammons Preparatory Academy Inc. | 850 MAIN ST, BUFFALO, NY | 0.06 |
| Regional Foundation LLC | 267 Leisure Lane, Burnsville, NC | Regional Fóundation LLC |  | 0.08 |
| Prime It LLP | Surendra Nagar, Gujarat, Wadhwancity, Patel Society, Nr Bhak | PRIME LLP IT |  | 0.20 |
| Witt and Hardesty LLC | 506 2nd Street, IN, Goshen | Witt and Hardesty |  | 0.17 |
| Mountain Initiative III Inc | 19305 Wagoner Road, Kirkland, AZ | ONYXLYRA | 19307 WAGONER ROAD, KIRKLAND, AZ | 0.65 |

### Examples: missed, one-to-one conflict
| S1 name | S1 address | candidate name | candidate address | p |
|---|---|---|---|---|
| Quezada Wave of Montville | 15991 Hart Road, Montville, OH | Orbilyra | 15991 Hart Road, Montville, OH | 0.14 |
| United Industries LLC | Town Of Bear Lake, 1675 29 1/4 Avenue, WI | United Industries Llc |  | 0.19 |
| Amar Tech Private Limited | Uttar Pradesh, 367, Ghaziabad, Sector-5 Vaishali, Ghaziabad | Verairi Co | 367, Sector-5 Vaishali, Ghaziabad, Uttar Pradesh | 0.07 |
| Taylor Royal Montage Group | 454 Stratford Drive, Henderson, NC | Taylor-Royal |  | 0.15 |
| FKO Agents Corp | Khasra No. 175, 170, 90/2, 176, Pipliya Dhakad, Huzur, Bhopa | FKO AGENTS | H.NO 175 , 170, 90/2, 176, PIPLIYA DHAKAD, HUZUR, मध्य प्रदे | 1.00 |
| Regional Ministries | 560 Peck Lane, Cheshire, CT | Regional Ministries Co |  | 0.02 |
| Board of Environmental Protection | 17439 Desmond Street, Humble, TX | BOARD OF ENVIRONMENTAL PROTECTION Enterprises |  | 0.00 |
| Silver Pilates | 2003 Mcknight Street, Humboldt, TN | Silver [Pilates] (ID: 28880) |  | 0.02 |
| Reid Lending PLLC | 632 Union Hill Road, Hendersonville, NC | Umbradelta | Union Hill Road, Hendersonville, North Carolina | 0.49 |
| FKO Agents Corp | Khasra No. 175, 170, 90/2, 176, Pipliya Dhakad, Huzur, Bhopa | Zetaavi Co | Khasra No. 175, 170, 90/2, 176, Pipliya Dhakad, Huzur, Bhopa | 0.26 |
| National Animal Hospital LLC | 23119 57, City Of Kiel, WI | National Animal Hospital Llc |  | 0.13 |
| Ram Business Private Limited | 90 Navin Nagar, Hinotiya, Huzur, Bhopal, Madhya Pradesh | Ram Business Private-Limited |  | 0.01 |
