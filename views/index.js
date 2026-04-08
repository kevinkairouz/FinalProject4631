async function fetchData(){   
    //TODO: will fix function and implement to send the data by using getElementByID and putting it in a varable 
    //and sending it to the server
    let response = await fetch("http://127.0.0.1:5000/test") 
    let data = response.json() 
    console.log(data)
    
    
    
    // console.log("HELLO IVE BEEN TAPPED")
    // return "" 
}

//TODO: work on the simple js function and print the result to the console