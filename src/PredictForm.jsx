import React, { useState } from "react";

export default function DropoutPredictor() {
  const [formData, setFormData] = useState({
    Gender: "",
    Education_Level: "",
    Course_Name: "",
    Engagement_Level: "",
    Learning_Style: "",
    Age: "",
    Time_Spent_on_Videos: "",
    Quiz_Attempts: "",
    Quiz_Scores: "",
    Forum_Participation: "",
    Assignment_Completion_Rate: "",
    Feedback_Score: ""
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Convert numeric fields to numbers or default to 0
    const payload = { ...formData };
    Object.keys(payload).forEach((key) => {
      if (
        [
          "Age",
          "Time_Spent_on_Videos",
          "Quiz_Attempts",
          "Quiz_Scores",
          "Forum_Participation",
          "Assignment_Completion_Rate",
          "Feedback_Score"
        ].includes(key)
      ) {
        payload[key] = payload[key] === "" ? 0 : Number(payload[key]);
      }
    });

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Prediction error:", error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <form
        onSubmit={handleSubmit}
        className="bg-white p-6 rounded-2xl shadow-lg grid grid-cols-2 gap-4 max-w-3xl w-full"
      >
        {/* Categorical Fields */}
        <input type="text" name="Gender" placeholder="Gender" value={formData.Gender} onChange={handleChange} className="border p-2 rounded" />
        <input type="text" name="Education_Level" placeholder="Education Level" value={formData.Education_Level} onChange={handleChange} className="border p-2 rounded" />
        <input type="text" name="Course_Name" placeholder="Course Name" value={formData.Course_Name} onChange={handleChange} className="border p-2 rounded" />
        <input type="text" name="Engagement_Level" placeholder="Engagement Level" value={formData.Engagement_Level} onChange={handleChange} className="border p-2 rounded" />
        <input type="text" name="Learning_Style" placeholder="Learning Style" value={formData.Learning_Style} onChange={handleChange} className="border p-2 rounded" />

        {/* Numeric Fields */}
        <input type="number" name="Age" placeholder="Age" value={formData.Age} onChange={handleChange} className="border p-2 rounded" />
        <input type="number" name="Time_Spent_on_Videos" placeholder="Time Spent on Videos" value={formData.Time_Spent_on_Videos} onChange={handleChange} className="border p-2 rounded" />
        <input type="number" name="Quiz_Attempts" placeholder="Quiz Attempts" value={formData.Quiz_Attempts} onChange={handleChange} className="border p-2 rounded" />
        <input type="number" name="Quiz_Scores" placeholder="Quiz Scores" value={formData.Quiz_Scores} onChange={handleChange} className="border p-2 rounded" />
        <input type="number" name="Forum_Participation" placeholder="Forum Participation" value={formData.Forum_Participation} onChange={handleChange} className="border p-2 rounded" />
        <input type="number" name="Assignment_Completion_Rate" placeholder="Assignment Completion Rate" value={formData.Assignment_Completion_Rate} onChange={handleChange} className="border p-2 rounded" />
        <input type="number" name="Feedback_Score" placeholder="Feedback Score" value={formData.Feedback_Score} onChange={handleChange} className="border p-2 rounded" />

        <button type="submit" className="col-span-2 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition">
          Predict
        </button>
      </form>

      {result && (
        <div className="mt-6 bg-white p-6 rounded-2xl shadow-md max-w-lg w-full">
          <p className="text-lg font-bold">
            Prediction: {result.prediction === 1 ? "High Dropout Risk" : "Low Dropout Risk"}
          </p>
          <p className="text-gray-600">
            Probability: {(result.probability * 100).toFixed(2)}%
          </p>

          {result.suggestions && result.suggestions.length > 0 && (
            <>
              <h3 className="mt-4 font-semibold text-lg">Actionable Suggestions:</h3>
              <ul className="list-disc ml-5 mt-2 space-y-1">
                {result.suggestions.map((sugg, index) => (
                  <li key={index}>{sugg}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}
