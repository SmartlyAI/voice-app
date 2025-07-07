let mediaRecorder;
let audioChunks = [];
let audioBlob;
let startTime;
let durationSec = 0;

const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const playBtn = document.getElementById('play');
const validerBtn = document.getElementById('valider');
const audioPlayer = document.getElementById('audio');
const successMessage = document.getElementById('success-message'); 

// Vérifier si les éléments existent avant d'ajouter des écouteurs d'événements
if (startBtn && stopBtn && playBtn && validerBtn && audioPlayer && successMessage) {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.onstart = () => {
          audioChunks = [];
          startTime = Date.now();
          successMessage.style.display = 'none'; // Cacher au début de l'enregistrement
        };

        mediaRecorder.ondataavailable = event => {
          audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
          durationSec = (Date.now() - startTime) / 1000;
          audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
          const audioURL = URL.createObjectURL(audioBlob);
          audioPlayer.src = audioURL;
          playBtn.disabled = false;
          validerBtn.disabled = false;
        };

        startBtn.onclick = () => {
          if (mediaRecorder.state === "inactive") {
            mediaRecorder.start();
            startBtn.disabled = true;
            stopBtn.disabled = false;
            playBtn.disabled = true; // Désactiver play pendant l'enregistrement
            validerBtn.disabled = true; // Désactiver valider pendant l'enregistrement
            audioPlayer.src = ''; // Vider le player
          }
        };

        stopBtn.onclick = () => {
          if (mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            startBtn.disabled = false;
            stopBtn.disabled = true;
            // playBtn et validerBtn seront activés dans mediaRecorder.onstop
          }
        };

        playBtn.onclick = () => {
          if (audioPlayer.src && audioPlayer.src !== window.location.href) { // S'assurer qu'il y a une source valide
            audioPlayer.play();
          }
        };

        validerBtn.onclick = () => {
          if (!audioBlob) {
            alert("Veuillez d'abord enregistrer l'audio.");
            return;
          }
          const formData = new FormData();
          formData.append('audio_data', audioBlob, 'recording.wav'); // Nom de fichier optionnel pour le blob
          formData.append('duration', durationSec.toFixed(2));
          formData.append('tifinagh', document.getElementById('tifinagh').value);
          formData.append('latin', document.getElementById('latin').value);

          validerBtn.disabled = true; // Désactiver pendant l'envoi
          validerBtn.textContent = 'Validation...';


          fetch('/upload', {
            method: 'POST',
            body: formData
          })
          .then(response => {
            if (!response.ok) {
              // Si le serveur renvoie une erreur (4xx, 5xx)
              return response.json().then(errData => {
                throw new Error(errData.message || `Erreur HTTP ${response.status}`);
              });
            }
            return response.json(); // Si OK (2xx)
          })
          .then(data => {
            if (data.success) {
              successMessage.textContent = data.message || '✅ Enregistrement effectué avec succès ! / تم التسجيل بنجاح!';
              successMessage.style.display = 'block';
              setTimeout(() => {
                window.location.href = "/recorder"; // Redirige vers /recorder, qui affichera la phrase correcte
              }, 1500); // Délai un peu plus long pour voir le message
            } else {
              // Gérer les erreurs métier renvoyées dans un JSON avec success: false
              alert("Erreur lors de la validation: " + (data.message || "Erreur inconnue."));
              validerBtn.disabled = false;
              validerBtn.innerHTML = 'Valider / <span dir="rtl">تأكيد</span>';
            }
          })
          .catch(error => {
            console.error('Erreur lors de la validation:', error);
            alert("Erreur lors de la validation: " + error.message);
            validerBtn.disabled = false;
            validerBtn.innerHTML = 'Valider / <span dir="rtl">تأكيد</span>';
          });
        };
      })
      .catch(err => {
        console.error("Erreur d'accès au microphone:", err);
        alert("Impossible d'accéder au microphone. Veuillez vérifier les permissions.");
        // Désactiver les boutons si le micro n'est pas accessible
        if(startBtn) startBtn.disabled = true;
        if(stopBtn) stopBtn.disabled = true;
        if(playBtn) playBtn.disabled = true;
        if(validerBtn) validerBtn.disabled = true;
      });
} else {
    console.warn("Certains éléments du DOM pour l'enregistreur audio n'ont pas été trouvés. La fonctionnalité d'enregistrement pourrait ne pas marcher.");
}