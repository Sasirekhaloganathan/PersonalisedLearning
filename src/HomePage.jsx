import React from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';

function HomePage() {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/predict');
  };

  return (
    <div className="home-container">
      <div className="home-content">
        <div className="hero-section">
          
          <h1 className="hero-title">
            Student Risk Prediction System
          </h1>
          
          <p className="hero-subtitle">
            Advanced AI-powered analytics to predict student performance and provide 
            personalized learning recommendations for academic success.
          </p>
          
          <button 
            className="cta-button"
            onClick={handleGetStarted}
          >
            <span className="button-icon">🚀</span>
            Start Prediction
            <span className="button-arrow">→</span>
          </button>
        </div>
      </div>
      
      <div className="background-elements">
        <div className="bg-circle circle-1"></div>
        <div className="bg-circle circle-2"></div>
        <div className="bg-circle circle-3"></div>
      </div>
    </div>
  );
}

export default HomePage;