import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Home from './Home';
import About from './About';
import Contact from './Contact';
import './App.css';
import { FaSun, FaMoon } from 'react-icons/fa';

export default function App() {
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    document.body.style.backgroundColor = darkMode ? '#121212' : '#ffffff';
    document.body.style.color = darkMode ? '#e0e0e0' : '#000000';
  }, [darkMode]);

  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };

  return (
    <Router>
      <div className='navbar'>
        <h2>Deepfake Detection</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <Link to='/'>Home</Link>
          <Link to='/about'>About</Link>
          <Link to='/contact'>Contact</Link>
          <span className='theme-toggle' onClick={toggleTheme}>
            {darkMode ? <FaSun /> : <FaMoon />}
          </span>
        </div>
      </div>

      <Routes>
        <Route path='/' element={<Home />} />
        <Route path='/about' element={<About />} />
        <Route path='/contact' element={<Contact />} />
      </Routes>

      <div className='footer'>
        <p>&copy; 2023 Deepfake Detection System</p>
      </div>
    </Router>
  );
}
