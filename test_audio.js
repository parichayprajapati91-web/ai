const fs = require('fs');
const path = require('path');

const audioPath = path.join(__dirname, 'client/public/sample voice 1.mp3');

if (fs.existsSync(audioPath)) {
  const audioBuffer = fs.readFileSync(audioPath);
  const base64Audio = audioBuffer.toString('base64');

  console.log(`File size: ${audioBuffer.length} bytes`);
  console.log(`Base64 length: ${base64Audio.length} characters`);
  console.log(`Base64 preview: ${base64Audio.substring(0, 50)}...`);

  // Save to file for testing
  fs.writeFileSync('test_audio_base64.txt', base64Audio);
  console.log('Base64 saved to test_audio_base64.txt');

  // Test the API
  const http = require('http');

  const postData = JSON.stringify({
    audioFormat: "mp3",
    audioBase64: base64Audio,
    language: "English"
  });

  const options = {
    hostname: 'localhost',
    port: 5000,
    path: '/api/voice-detection',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': 'DEMO-KEY-123',
      'Content-Length': Buffer.byteLength(postData)
    }
  };

  const req = http.request(options, (res) => {
    console.log(`Status: ${res.statusCode}`);
    console.log(`Headers:`, res.headers);

    res.setEncoding('utf8');
    let body = '';
    res.on('data', (chunk) => {
      body += chunk;
    });
    res.on('end', () => {
      console.log('Response:', body);
    });
  });

  req.on('error', (e) => {
    console.error(`Problem with request: ${e.message}`);
  });

  req.write(postData);  req.end();
} else {
  console.error(`Audio file not found: ${audioPath}`);
}