# Manuale Utente FrameShift GUI

## 1. Introduzione

Benvenuto a FrameShift GUI! Questa applicazione fornisce un'interfaccia grafica facile da usare per lo strumento FrameShift, che permette di modificare automaticamente il rapporto d'aspetto dei video (ad esempio, da orizzontale a verticale) mantenendo i soggetti importanti nell'inquadratura.

Con questa GUI, puoi accedere alle potenti funzionalità di reframing di FrameShift senza dover utilizzare la riga di comando.

## 2. Requisiti

**Se esegui da codice sorgente:**

*   **Python:** Versione 3.8 o successiva.
*   **Dipendenze di FrameShift:** Tutte le librerie Python elencate nel file `requirements.txt` del progetto FrameShift devono essere installate (es. `opencv-python`, `mediapipe`, `ultralytics`, `scenedetect`, `numpy`, `tqdm`). Puoi installarle con:
    ```bash
    pip install -r requirements.txt
    ```
*   **FFmpeg (Fortemente Consigliato):** Per processare e includere l'audio dal video originale nel video reframato.
    *   Scarica FFmpeg da [ffmpeg.org](https://ffmpeg.org/download.html).
    *   Installalo e assicurati che la directory contenente l'eseguibile `ffmpeg` (o `ffmpeg.exe`) sia aggiunta al PATH del tuo sistema.
    *   Se FFmpeg non è trovato, i video verranno processati senza audio e un avviso apparirà nei log.

**Se utilizzi un eseguibile standalone (se disponibile):**

*   L'eseguibile dovrebbe includere tutte le dipendenze Python necessarie.
*   Potrebbe comunque essere necessario installare FFmpeg separatamente e aggiungerlo al PATH del sistema per la gestione dell'audio, a meno che non sia specificato diversamente nella documentazione dell'eseguibile.

## 3. Come Avviare la GUI

**Da codice sorgente:**

1.  Assicurati di aver installato tutti i requisiti.
2.  Naviga nella directory principale del progetto FrameShift tramite un terminale o prompt dei comandi.
3.  Esegui il seguente comando:
    ```bash
    python frameshift_gui.py
    ```
    (Se `frameshift_gui.py` si trova in una sottocartella, adatta il percorso, ad esempio `python nome_sottocartella/frameshift_gui.py`).

**Da un eseguibile (se fornito):**

*   Fai doppio click sul file eseguibile (es. `FrameShiftGUI.exe` su Windows).

## 4. Descrizione dell'Interfaccia Utente

La GUI è organizzata in diverse sezioni:

*   **Input/Output:**
    *   **Input Video/Cartella:** Campo per visualizzare il percorso del video sorgente o della cartella (per elaborazione batch). Clicca "Sfoglia..." per selezionare.
    *   **Output Video/Cartella:** Campo per visualizzare il percorso di destinazione del video elaborato o della cartella. Clicca "Sfoglia..." per selezionare.
    *   **Processa in Batch:** Spunta questa casella per elaborare tutti i video presenti in una cartella di input e salvarli in una cartella di output.

*   **Impostazioni Principali:**
    *   **Rapporto d'Aspetto:** Il formato desiderato per il video finale (es. `9:16` per video verticali, `1:1` per quadrati, `16:9` per orizzontali).
    *   **Altezza Output (pixel):** L'altezza del video finale in pixel (es. `1080`, `720`). La larghezza verrà calcolata automaticamente.
    *   **Interpolazione:** L'algoritmo usato per ridimensionare i fotogrammi. `lanczos` e `cubic` offrono buona qualità, `area` è utile per rimpicciolire, `linear` è più veloce.

*   **Impostazioni Padding:**
    *   **Abilita Padding:** Se il video originale, una volta ritagliato, non riempie completamente il nuovo formato, questa opzione permette di aggiungere barre laterali invece di tagliare ulteriormente l'immagine.
    *   **Tipo di Padding:** (Attivo solo se "Abilita Padding" è spuntato)
        *   `black`: Aggiunge barre nere.
        *   `blur`: Aggiunge barre create sfocando i bordi del video originale.
        *   `color`: Aggiunge barre di un colore solido a tua scelta.
    *   **Intensità Sfocatura:** (Attivo per padding `blur`) Regola quanto devono essere sfocate le barre (0=minima, 10=massima).
    *   **Colore Padding:** (Attivo per padding `color`) Inserisci un nome di colore (es. `red`, `green`) o un codice RGB (es. `(255,0,0)`). Puoi anche cliccare "Scegli..." per selezionare un colore da una palette.

*   **Impostazioni Avanzate:**
    *   **Pesi Oggetti:** Controlla l'importanza data ai diversi tipi di oggetti rilevati (es. volti, persone) nel determinare il ritaglio. Formato: `etichetta1:peso1,etichetta2:peso2`. Pesi più alti danno maggiore priorità. `default` si applica a oggetti non specificati.
    *   **Opacità Contenuto:** Controlla la trasparenza del video principale. Se inferiore a 1.0, il video viene fuso con uno sfondo sfocato del frame originale. 1.0 è completamente opaco.

*   **Azioni e Stato:**
    *   **AVVIA PROCESSO:** Bottone per iniziare l'elaborazione video con le impostazioni correnti.
    *   **Barra di Progresso:** Mostra l'avanzamento dell'elaborazione.
    *   **Area di Log:** Visualizza messaggi informativi, avvisi ed errori durante il processo.

*   **Menu "Aiuto":**
    *   **Informazioni su FrameShift GUI:** Mostra dettagli sulla versione e sullo scopo dell'applicazione.
    *   **Guida Rapida (Placeholder):** Fornisce i passaggi essenziali per usare la GUI.

## 5. Guida Rapida all'Uso

1.  **Avvia FrameShift GUI.**
2.  **Seleziona Input:** Clicca "Sfoglia..." nella sezione Input e scegli il tuo file video. Se vuoi processare più video, spunta "Processa in Batch" e seleziona una cartella.
3.  **Seleziona Output:** Clicca "Sfoglia..." nella sezione Output e scegli dove salvare il video processato (o la cartella di output per il batch).
4.  **Configura Impostazioni:**
    *   Imposta il "Rapporto d'Aspetto" desiderato (es. `9:16` per TikTok/Instagram Stories).
    *   Regola l'"Altezza Output" se necessario.
    *   Se vuoi che il video riempia il frame tagliando i bordi, lascia "Abilita Padding" deselezionato. Se preferisci vedere l'intero contenuto ritagliato con l'aggiunta di barre, spunta "Abilita Padding" e scegli il tipo di padding.
    *   Esplora le "Impostazioni Avanzate" se hai esigenze specifiche.
5.  **Controlla i Tooltip:** Passa il mouse sopra le varie opzioni per visualizzare suggerimenti utili.
6.  **Avvia:** Clicca il bottone "AVVIA PROCESSO".
7.  **Attendi:** L'elaborazione potrebbe richiedere tempo, specialmente per video lunghi o in modalità batch. Puoi monitorare l'avanzamento nella barra di progresso e leggere i messaggi nell'area di log.
8.  **Fatto!** Una volta completato, troverai il video reframato nel percorso di output specificato.

## 6. Risoluzione Problemi Semplice / FAQ

*   **"FFmpeg non trovato" (Warning nel log):**
    *   Significa che FFmpeg non è installato o non è nel PATH del tuo sistema. Il video verrà processato senza audio. Per includere l'audio, installa FFmpeg e assicurati che sia accessibile dal terminale.

*   **Errori di importazione all'avvio (es. "No module named 'ultralytics'"):**
    *   Questo accade se stai eseguendo la GUI da codice sorgente e le dipendenze di FrameShift non sono installate correttamente. Segui le istruzioni nella sezione "Requisiti" per installarle (solitamente `pip install -r requirements.txt`).

*   **L'applicazione sembra bloccata o non risponde:**
    *   L'elaborazione video, specialmente per file lunghi o con analisi complesse, può richiedere molto tempo. La GUI esegue il lavoro pesante in un thread separato per rimanere il più possibile responsiva, ma operazioni intensive potrebbero comunque dare questa impressione. Controlla l'area di log per messaggi di attività o errori. Se la barra di progresso si muove (o è in modalità indeterminata attiva), probabilmente sta ancora lavorando.

*   **Modelli YOLO non trovati / Errori di download:**
    *   FrameShift usa modelli di intelligenza artificiale (YOLO) per rilevare volti e oggetti. Questi modelli vengono solitamente scaricati automaticamente la prima volta che vengono usati. Assicurati di avere una connessione internet attiva durante il primo avvio o se i modelli non sono presenti. Se ci sono problemi persistenti con i modelli, potrebbero esserci problemi di rete o con la cache dei modelli di `ultralytics`.

*   **Tooltip non chiari o mancanti:**
    *   I tooltip sono lì per aiutarti! Se un'opzione non è chiara, il tooltip potrebbe fornire il contesto necessario.

---
Speriamo che questa guida ti sia utile per utilizzare FrameShift GUI!
