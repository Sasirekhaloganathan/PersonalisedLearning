import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import fetch from "node-fetch";

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Proxy route: sends data to Python backend
app.post("/predict", async (req, res) => {
  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ error: "Prediction failed" });
  }
});

const PORT = 4000;
app.listen(PORT, () => {
  console.log(`🚀 Node backend running on http://localhost:${PORT}`);
});
