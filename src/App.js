import React, { useState } from 'react';
import './App.css';
import loadingGIF from './loading.gif'

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [timeTaken, setTimeTaken] = useState(null);
  const [realFrames, setRealFrames] = useState(null); // State for real frames
  const [fakeFrames, setFakeFrames] = useState(null); // State for fake frames
  const [analyzing, setAnalyzing] = useState(false);

  const handleVideoUpload = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleVideoSubmission = async () => {
    if (!selectedFile) {
      alert("Please select a video to upload!");
      return;
    }

    setAnalyzing(true);  // Start analyzing

    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:5000/upload_video', {  // Update Flask server URL
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      console.log("Response from server:", data);  // Log the response from the server

      setResult(data.result);
      setTimeTaken(data.time_taken);
      setRealFrames(data.real_frames); // Store the number of real frames
      setFakeFrames(data.fake_frames); // Store the number of fake frames

      setTimeout(() => {
        setAnalyzing(false);
      }, 1000);

    } catch (error) {
      console.error('There was an error uploading the video:', error);
      setAnalyzing(false);  // Stop analyzing in case of error
    } finally {
      setAnalyzing(false);  // End analyzing regardless of success or failure
    }
  };

  return (
    <>
      <div className='header'>
        <h1><i>"Deepfake Videos Detection System using Deep Learning"</i></h1>
      </div>

      <div className="upload-box">
        <div>
          <i style={{ fontSize: '35px' }} className="fa-solid fa-file"></i>
          <b style={{ fontSize: '30px' }}> Upload a video!</b>
        </div>

        <input
          className="file-input"
          type="file"
          accept="video/*"
          onChange={handleVideoUpload}
          style={{ marginLeft: '66px' }}
          disabled={analyzing} // Disable input while analyzing
        />

        <button className="submit-video" onClick={handleVideoSubmission} disabled={analyzing}>
          {analyzing ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      <div className="loadingGIF">
        {analyzing && (
          <>
            <h2>Searching For Anomalies</h2>
            <img src={loadingGIF} alt="Loading" />
          </>
        )}
      </div>

      {result && (
        <div className="result-box">
          <h3>Detection Results:</h3>
          <p><strong>The provided video seems </strong>
            <b style={{ color: result.includes("deepfake") ? 'red' : 'green' }}>
              "{result}"
            </b>
          </p>
          <p><strong>Time taken for detection: </strong>{timeTaken}</p>
          <p><strong>Real Frames: </strong>{realFrames}</p> {/* Display real frames */}
          <p><strong>Fake Frames: </strong>{fakeFrames}</p> {/* Display fake frames */}
        </div>
      )}
    </>
  );
}
