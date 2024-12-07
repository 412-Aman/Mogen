async function searchMovie() {
    const userId = 1; // Replace with user ID
    const feedback = ""; // Optionally add feedback
    const response = await fetch('http://127.0.0.1:5000/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, feedback: feedback })
    });
    const data = await response.json();
    displayRecommendations(data.recommendations);
}

function displayRecommendations(recommendations) {
    const recList = document.getElementById('recommendationsList');
    recList.innerHTML = '';
    recommendations.forEach(movie => {
        const div = document.createElement('div');
        div.textContent = movie;
        recList.appendChild(div);
    });
}