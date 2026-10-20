# ÁTADÓ: YuE2 szerver LAN-elérése a 3090-es Windowson

> Ezt a fájlt a 3090-es gépen futó Claude Code session olvassa el először.
> A munka a fejlesztői laptopon (Ryzen AI 9 365, 192.168.50.12) folyik, ez a fájl viszi át a kontextust.

## Munkaszabályok (kötelező)
- **Mindig magyarul kommunikálj.**
- **Terv + kifejezett „igen” minden fájlmódosítás és rendszerbeállítás előtt** (Edit/Write, .bat írása,
  tűzfalszabály, szolgáltatás). Olvasni, diagnosztizálni, parancsot futtatni a helyzet felméréséhez
  engedély nélkül is szabad. A felhasználó ezt többször, nyomatékosan kérte.
- **„Teszteld” = technikai ellenőrzés:** mi fut, milyen beállítással, válaszol-e a végpont, mennyi
  ideig tart. Zenei minőséget ne értékelj.
- Ne bonyolítsd túl. Az eredeti, CUDA-s yue2-studio (`sanyi1963bp/yue2-studio`) már fel van téve
  erre a gépre, és működik. Csak azt csináld, ami lent szerepel.

## A cél
A laptopon futó yue2-studio (fork: `Marhivo/yue2-studio`, „távoli szerver” móddal) **ennek a gépnek az
`audiocpp_server`-ét használja** a LAN-on. A GUI a laptopon marad, a számolás (YuE2, HTDemucs,
SheetSage2, MuScriptor, Qwen3-ASR) itt fut CUDA-n.

A laptop oldala kész és tesztelve van:
- A laptop elküldi a modell betöltési kérését **ennek a gépnek a modellmappájával** (ezt a felhasználó
  adja meg a laptopon).
- A bemeneti WAV-okat a `POST /v1/ui/upload` végponton tölti fel (ehhez a szervert `--ui` kapcsolóval
  kell indítani).
- Az eredményt base64-ben kapja vissza.

## Teendők ezen a gépen
1. **Mérd fel a helyzetet (engedély nélkül):**
   - Hol van az eredeti klón, a motor (`audiocpp_server.exe`) és a `model_specs` mappa?
     Az app munkakönyvtára a `motor` mappa, ha abban van `model_specs`, különben az exe mappája.
   - Mi a `modellek` mappa pontos útvonala?
   - Mi a gép IPv4-címe (várhatóan 192.168.50.122)?
   - `nvidia-smi`: driver- és VRAM-állapot.
2. **Indító `.bat` (terv + igen után):** `lan_szerver.bat` az eredeti klón gyökerébe. Feladatai:
   - Indulás előtt törli a korábbi feltöltések maradékát: `for /d %%D in ("%TEMP%\audiocpp-ui-*") do rd /s /q "%%D"`.
     A szerver minden indításkor létrehoz egy ilyen mappát, és a feltöltött WAV-ok is ide kerülnek.
   - A helyes munkakönyvtárból (lásd 1. pont) elindítja:
     `audiocpp_server.exe --ui --ui-management --backend cuda --host 0.0.0.0 --port 8080 --busy-timeout-ms 0`
   - A `--busy-timeout-ms 0` kell, mert a hosszú generálás 10 percnél is tovább tarthat.
3. **Tűzfal (terv + igen után, rendszergazdaként), MÉG AZ ELSŐ INDÍTÁS ELŐTT:**
   - Program- és címszintű szabály:
     `netsh advfirewall firewall add rule name="YuE2 audiocpp (laptop)" dir=in action=allow program="<audiocpp_server.exe teljes útvonala>" protocol=TCP localport=8080 remoteip=192.168.50.12`
   - Ha az első indításkor mégis feljön a Windows „Engedélyezés” ablaka, **ne** a „Mégse” gombot nyomd.
     A Mégse letiltó szabályt hoz létre a programra, és az felülírja az engedélyezőt.
   - Érdemes jelezni a felhasználónak: a laptop Wi-Fin DHCP-vel kapja a címét. Ha a cím megváltozik, a
     szabály nem fog működni, ezért a routerben érdemes fix címet foglalni a laptopnak.
4. **Ellenőrzés:** a `.bat` indítása után ezen a gépen a `curl http://127.0.0.1:8080/health` válaszában
   `"backend":"cuda"` szerepeljen.
5. **Visszajelzés a felhasználónak** (ő viszi át a laptopra):
   - a gép IP-címe,
   - a **`modellek` mappa teljes útvonala** (perjellel is jó, pl. `C:/.../yue2-studio/modellek`),
   - a `/health` válasza.

## Fontos tudnivalók
- **A motor verziója maradjon v0.8.1** (az eredeti app ezt tölti le). A laptop ugyanezt használja, az
  API-t erre teszteltük.
- **Ne használja egyszerre a szervert a 3090 saját GUI-ja és a laptop.** Mindkettő kiüríti a „másik”
  modellt betöltés előtt, és egymás alól töltenék ki a modelleket.
  Ha ezen a gépen elindul az eredeti app, észleli a már futó szervert, és nem indít újat. Ez rendben van.
- A szervernek **nincs hitelesítése**, és a `--ui-management` miatt bármilyen szerveroldali fájlt be tud
  tölteni. Ezért fontos, hogy a tűzfalszabály csak a laptop címére szóljon.
- A SheetSage2 itt CUDA-n fut. A laptopon Vulkanon nem fér el (audio.cpp#996), ezért ott CPU-n futott.
