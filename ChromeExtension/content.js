var HttpClient = function() {
    this.get = function(aUrl, aCallback) {
        var anHttpRequest = new XMLHttpRequest();
        anHttpRequest.onreadystatechange = function() { 
            console.log(anHttpRequest.status)
            if (anHttpRequest.readyState == 4 && anHttpRequest.status == 200)
                aCallback(anHttpRequest.responseText);
        }

        anHttpRequest.open( "GET", aUrl, true );            
        anHttpRequest.send( null );
    }
}


setInterval(() => {
    console.log('Running now')
    var game_names = document.getElementsByTagName('a');
    
    for (var i = 0; i < game_names.length; i++) {
        if(!game_names[i].hasAttribute('game_rating')){
            if (game_names[i].dataset.test === "product-title") {
                game_names[i].innerHTML = game_names[i].innerHTML + "<br /> Game complexity is blahh!";
                game_names[i].setAttribute('game_rating', 'Done');
                var client = new HttpClient();
                
                client.get('http://127.0.0.1:5000/input', response =>
                 {console.log(response)
                 });
                
            }
        }
    }
}, 5000)
