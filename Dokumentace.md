# Ročníkový projekt - Dron

## Dokumentace

Cílem projektu je příprava dat pro machine learning modely a  implementace některých takových modelů pro konkrétní data. Projekt by měl umožňovat zkoušet různé přístupy k předzpracování dat (délka intervalu, reprezentant intervalu, funkce, kterými jsou hodnoty z intervalu zpracovány) a poskytovat tak uživateli možnost nalézt optimální způsob předzpracování dat tak, aby byl následný machine learning co nejefektivnější.

Projekt pracuje s daty o letu dronu. Na základě dat o poloze dronu se pokouší předpovídat, jaké příkazy dron dostal. Data o poloze jsou čtena z `.tsv` souboru v následujícím formátu:

```xml
<toplevelgroup name="Navdata" file="navdata.tsv" separator="tab">
    <time unit="milliseconds" source="recorder" />
    <intEnumeration name="State">
        <option value="0" label="Unknown" />
        <option value="1" label="Init" />
        <option value="2" label="Landed" />
        <option value="3" label="Flying" />
        <option value="4" label="Hovering" />
        <option value="5" label="Test" />
        <option value="6" label="Taking off" />
        <option value="7" label="Go to point" />
        <option value="8" label="Landing" />
        <option value="9" label="Looping" />
    </intEnumeration>
    <intQuantity name="Battery level" unit="percent"/>
    <group name="Magnetometer readings">
        <!-- Unit: might be milli-g ?-->
        <intQuantity name="Magnetometer x" unit="?" />
        <intQuantity name="Magnetometer y" unit="?" />
        <intQuantity name="Magnetometer z" unit="?" />
    </group>
    <intQuantity name="Pressure" unit="Pa"/>
    <intQuantity name="Temperature" unit="?"/>
    <group name="Wind conditions">
        <doubleQuantity name="Wind speed" unit="m/s" />
        <!-- Estimated wind direction in North-East frame [rad]  e.g. if wind_angle is pi/4, wind is from South-West to North-East -->
        <!-- Is the documentation right? It seems like that the unit is in fact degrees. -->
        <doubleQuantity name="Wind angle" unit="rad" />
        <doubleQuantity name="Wind compensation: pitch" unit="milli-degrees" />
        <doubleQuantity name="Wind compensation: roll" unit="milli-degrees" />
    </group>
    <group name="Rotation">
        <doubleQuantity name="Pitch (Rotation in y)" unit="milli-degees" />
        <doubleQuantity name="Roll (Rotation in x)" unit="milli-degees" />
        <doubleQuantity name="Yaw (Rotation in z)" unit="milli-degees" />
    </group>
    <intQuantity name="Altitude" unit="millimeters" />
    <group name="Velocity">
        <doubleQuantity name="Velocity in x" unit="mm/s"/>
        <doubleQuantity name="Velocity in y" unit="mm/s"/>
        <doubleQuantity name="Velocity in z" unit="mm/s"/>
    </group>
    <group name="Acceleration">
        <doubleQuantity name="Acceleration in x" unit="mg"/>
        <doubleQuantity name="Acceleration in y" unit="mg"/>
        <doubleQuantity name="Acceleration in z" unit="mg"/>
    </group>
    <group name="Motor powers">
        <intQuantity name="Motor 1 power" value="0-255"/>
        <intQuantity name="Motor 2 power" value="0-255"/>
        <intQuantity name="Motor 3 power" value="0-255"/>
        <intQuantity name="Motor 4 power" value="0-255"/>
    </group>
    <time unit="microseconds" source="board"/>
</toplevelgroup>
```

Data o příkazech jsou získána z `tsv.` souboru ve formátu:

```xml
<toplevelgroup name="Commands" file="commands.tsv" separator="tab">
    <time unit="milliseconds" source="recorder" />
    <doubleQuantity name="Left-right tilt" unit="-1 to 1" />
    <doubleQuantity name="Front-back tilt" unit="-1 to 1" />
    <doubleQuantity name="Vertical speed" unit="-1 to 1" />
    <doubleQuantity name="Angular speed" unit="-1 to 1" />
</toplevelgroup>
```

Projekt se skládá z 7 modulů v programovacím jazyce Python.

### `ReadResampleMerge`

Tento modul a stejnojmenná metoda v něm obsažená umožňuje načíst data o poloze dronu (`Navdata`) a příkazy, které dron dostal (`Commands`) ze  dvou  `tsv.` souborů odpovídajícího formátu. Data následně převzorkuje tak, že mezi každými dvěma po sobě jdoucími záznamy je fixní časový rozestup (defaultně 50 milisekund) a slije oba dva soubory do jednoho podle času (tedy data o poloze jsou spojena s příkazem, který dron v daném okamžiku dostal). Dostane-li metoda jako volitelé parametry jména souborů, vypíše do nich podobu dat v jednotlivých fázích zpracování. Je možné vypsat převzorkovaná `Navdata`, převzorkované `Commands` a výsledný soubor po slití obou dílčích souborů dohromady.

Příklad volání metody:

```python
# prints resapmpled Navdata and Commands and merged result into corresponding files
merged = ReadResampleMerge('InputData\\commands.tsv', 'InputData\\navdata.tsv', '50ms', 'OutputStages\\resampledCmds.tsv',
                           'OutputStages\\resampledNav.tsv', "OutputStages\\mergedResampled.tsv")

# does not output data into files
mergedSecondSet = ReadResampleMerge('InputData\\cmdsSecondSet.tsv', 'InputData\\navdataSecondSetTABS.tsv', '50ms')
```

[resampledCMDs](https://github.com/prusovak2/Drone/blob/master/OutputStages/resampledCmds.tsv)

[resampledNavdata](https://github.com/prusovak2/Drone/blob/master/OutputStages/resampledNav.tsv)

[merged](https://github.com/prusovak2/Drone/blob/master/OutputStages/mergedResampled.tsv)

### `CreateDataMatrix`

V tomto modulu se odehrává hlavní část předzpracování dat. Zmergovaná data (tedy výstup metody `ReadResampleMerge`) jsou rozdělena do intervalů po určitém počtu záznamů  (default 40 záznamů na interval), z každého intervalu je vybrán jeden záznam jako reprezentant, jehož čas a `Comands` následně určují interval (defaultně první záznam z intervalu) a z hodnot v některých z ostatních sloupců jsou  pak konkrétními funkcemi spočítány hodnoty `features` do výsledné matice, na které se budou učit machine learning modely (ke které je jak v dokumentačních komentářích ve zdrojovém kódu, tak ve zbytku této dokumentace odkazováno jako k `dataMatrix`). Defaultní funkce jsou `numpy.mean`, `numpy.std`, a `numpy.fft` z jejich výsledků. Hodnoty jsou počítány ze sloupců `Roll`, `Pitch` a `Yaw`, které nesou informace o rotaci dronu po řadě v osách x, y a z. 

#### Průběh zpracování dat

Nejprve je metodou `CreateEmptyDataFrame` (resp. `CreateEmpyDataFrameWithShift`) vytvořen prázdný `DataFrame` (`pandas.DataFrame`) o rozměrech, které odpovídají výsledné `dataMatrix`. Počet sloupců je určen jmény sloupců, jež jsou předána této metodě. Počet řádek je dán tím, jaká je zvolena délka intervalu (tedy kolik záznamů ze zmergovaných dat má být slito do jednoho záznamu v `dataMatrix`). V rámci této metody je také zvolen reprezentant intervalu (určuje časový index záznamu). Metodě je třeba předat `DataFrame` oindexovaný intovými hodnotami (`intIndexed = inputDFmerged.reset_index()`).

Následně je `DataFrame` vyplněn daty. Metoda `MakeCMDsDiscreteWithFrozenDict` reprezentuje hodnoty příkazů odpovídajícími členy následujícího `dictionary`:

 ```python
frozenCmds = frozendict({1: '-', 2: '0', 3: '+'})
 ```

Některou z `CreateData` metod jsou hodnoty z konkrétních sloupců merged dat zpracovány konkrétními funkcemi do `features` výsledné `dataMatrix`.

Celý tento postup je vykonán v rámci volání metody `CreateDataFrameForDTMatrix` (resp. `CreateDataFrameForDTMatrixShift`).

Příklad volání:

```python
dataColumnNamesRealImag = ['leftRight', 'frontBack', 'angular', 'Roll_Mean', 'Roll_SD', 'Roll_FFT_Mean_Real', 'Roll_FFT_Mean_Imag',
                           'Roll_FFT_SD', 'Pitch_Mean', 'Pitch_SD', 'Pitch_FFT_Mean_Real', 'Pitch_FFT_Mean_Imag', 'Pitch_FFT_SD',
                           'Yaw_Mean', 'Yaw_SD', 'Yaw_FFT_Mean_Real', 'Yaw_FFT_Mean_Imag', 'Yaw_FFT_SD']

dataForCM = CreateDataFrameForDTMatrix(inputDFmerged=mergedCM, ColumnNames=dataColumnNamesRealImag, intervalLen=40)
```

#### Úpravy předzpracování dat

Projekt má umožňovat hledání nejlepšího způsobu, jak připravit data `Navdata` a `Commands` pro machine learning. Za tímto účelem metoda `CreateDataFrameForDTMatrixShift` a v rámci jejího volání metody `CreateEmpyDataFrameWithShift` a `CreateData` poskytují možnost nastavit některé aspekty předzpracování dat. Podívejme se na signaturu této metody.

```python
def CreateDataFrameForDTMatrixShift(inputDFmerged, intervalLen=40, representantSampleShift=0, funcArrayToCreateContent=None, ColumnNames=None):
```

Parametr `intervalLen` určuje, kolik záznamů ve zmergovaných datech má tvořit interval, ze kterého je spočítán jeden záznam ve výsledné `dataMatrix`. Parametr `representantSampleShift` říká, kolikátý záznam z intervalu má poskytnout čas a hodnoty `Commands` pro záznam v `dataMatrix` vytvořený na základě tohoto intervalu. 

Zajímavým parametrem je `funcArrayToCreateContent`. Ten umožňuje nastavit, jakými funkcemi mají být data z intervalu zpracována do `features` výsledné `dataMatrix` (místo defaultních `mean`, `std`,`fft(mean)`, `fft(std)` ze sloupců `Roll`, `Pitch` a `Yaw`). Jako tento parametr musí být předáno pole tříčlenných `Dictionaries`. Každé dílčí `Dictionary` určuje jeden `feature` sloupec výsledné `dataMatrix`. `Dictionary` musí obsahovat klíče `'Name'`, `'Func'`, a  `'Column'`. Name určuje jméno sloupce v `dataMatrix`, do kterého má být hodnota vyplněna. Musí odpovídat některému ze jmen sloupců v `ColumnNames`. Hodnota `Func` má být delegát na funkci, kterou jsou hodnoty z intervalu zpracovány do jedné hodnoty v `dataMatrix`. `Column` říká, z jakého sloupce ze vstupního (zmergovaného) `dataFrame` mají být argumenty pro funkci brány.
Příklad, jak může hodnota argumentu `funcArrayToCreateContent` vypadat:

```python
namesAndFunc = [{'Name': 'Roll_Mean', 'Func': np.mean, 'Column': 'Roll_x'}, {'Name': 'Roll_SD', 'Func': np.std, 'Column': 'Roll_x'},
                {'Name': 'Pitch_Mean', 'Func': np.mean, 'Column': 'Pitch_y'}, {'Name': 'Pitch_SD', 'Func': np.std, 'Column': 'Pitch_y'},
                {'Name': 'Yaw_Mean', 'Func': np.mean, 'Column': 'Yaw_z'}, {'Name': 'Yaw_SD', 'Func': np.std, 'Column': 'Yaw_z'}]
```

Příklad volání metody:

```python
funcParamsData = CreateDataFrameForDTMatrixShift(inputDFmerged=merged, intervalLen=40, representantSampleShift=0,
                                                 funcArrayToCreateContent=namesAndFunc, ColumnNames=dataColumnNamesFucParams)
                                                 funcArrayToCreateContent=namesAndFunc, ColumnNames=dataColumnNamesFucParams)
funcParamsData.to_csv('OutputStages\\funcParamData.tsv', sep='\t')
```

[Výsledný soubor](https://github.com/prusovak2/Drone/blob/master/OutputStages/funcParamData.tsv)

Tento způsob určování funkcí pro zpracování dat z intervalů do `feature` hodnot má své omezení v tom, že umožňuje jako argumenty funkci předat pouze data přímo ze zmergovaného `dataFrame`. Není tedy možno hodnotu ve sloupci založit na hodnotě jiného sloupce v `dataMatrix` (např. počítat `fft(mean)`). Pro složitější způsoby počítání hodnot ve sloupcích `dataMatrix` doporučuji uživateli si napsat vlastní metodu podobnou metodě `CreateDataWithRealAndImagPart` (ve které určí, jak se mají sloupce vyplňovat) a tu pak předat metodě `CreateDataFrameForDTMatrix` jako argument `functionToCreateContent`.

### Machine learning modely

 