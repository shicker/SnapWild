import React, { useRef, useState } from 'react';

function App() {
  const videoRef = useRef(null);
  const [animal, setAnimal] = useState(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
    } catch (err) {
      console.error('Error accessing camera:', err);
    }
  };

  const captureAndPredict = async () => {
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const image = canvas.toDataURL('image/jpeg');

    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: JSON.stringify({ image }),
      headers: { 'Content-Type': 'application/json' },
    });
    const data = await response.json();
    setAnimal(data.animal);
  };

  return (
    <div>
      <h1>Snapwild</h1>
      <button onClick={startCamera}>Open Camera</button>
      <button onClick={captureAndPredict}>Recognize Animal</button>
      <video ref={videoRef} autoPlay style={{ width: '100%', maxWidth: '500px' }}></video>
      {animal && <h2>Detected Animal: {animal}</h2>}
    </div>
  );
}

export default App;