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

   
function callOtherDomain() {
  if(invocation) {    
    invocation.open('GET', url, true);
    invocation.onreadystatechange = handler;
    invocation.send(); 
  }
}

setInterval(() => {
    console.log('Running now')
    var game_names = document.getElementsByTagName('a');
    
    for (var i = 0; i < game_names.length; i++) {
        if(!game_names[i].hasAttribute('game_rating')){
            if (game_names[i].dataset.test === "product-title") {
                var client = new HttpClient();
                
                client.get('http://127.0.0.1:5000', {game_name: game_names[i]}, response =>
                 {console.log(response)
                 });
            
                 game_names[i].innerHTML = game_names[i].innerHTML + "<br /> Game complexity is ";
                 game_names[i].setAttribute('game_rating', 'Done');
            }
        }
    }
}, 5000)
