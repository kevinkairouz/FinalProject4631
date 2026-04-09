async function fetchData(){   
    let age = document.getElementById("Age").value 
    let screen_time = document.getElementById("ScreenTime").value 
    let social_hours = document.getElementById("SocialHours").value 
    let study_hours = document.getElementById("StudyHours").value 
    let sleep_hours = document.getElementById("SleepHours").value 
    let notis = document.getElementById("Noti").value  
    
    obj = {
        "age": age, 
        "ScreenTime": screen_time, 
        "SocialHours": social_hours,
        "StudyHours": study_hours, 
        "SleepHours": sleep_hours, 
        "Noti": notis 
    }
    
    let prediction = await fetch("http://127.0.0.1:5000/predict",{method: "POST", headers: {"Content-type": "application/json"}, body: JSON.stringify(obj)}) 
    let data = await prediction.json()
    console.log(data); 
    return "SUCCESS"; 
}

