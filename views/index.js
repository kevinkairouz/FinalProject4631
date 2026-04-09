async function fetchData(){   
    let age = document.getElementById("Age").value 
    let screen_time = document.getElementById("ScreenTime").value 
    let social_hours = document.getElementById("SocialHours").value 
    let study_hours = document.getElementById("StudyHours").value 
    let sleep_hours = document.getElementById("SleepHours").value 
    let notis = document.getElementById("Noti").value  

    //TODO: fix on sending the data to the server to get the response 
    //back and print it to the console
    let prediction = await fetch("http://127.0.0.1:5000/predict")
    
    console.log(prediction); 
    return "SUCCESS"; 
}

