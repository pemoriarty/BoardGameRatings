
function receivefunc(mssg,sender,sendResponse){
if(mssg.txtt==="execute"){


var HttpClient = function() {
    this.get = function(aUrl, aCallback, element) {
        var anHttpRequest = new XMLHttpRequest();
        anHttpRequest.onreadystatechange = function() { 
            //console.log(anHttpRequest.status)
            if (anHttpRequest.readyState == 4 && anHttpRequest.status == 200)
                aCallback(anHttpRequest.responseText, element);
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
            game_names[i].setAttribute('game_rating', 'Done')
            if (game_names[i].dataset.test === "product-title") {
                
                var client = new HttpClient();
                var temp = game_names[i];
                client.get(
                    'https://gameonapp.xyz/weight?game_name=' + temp.innerHTML, 
                    (response, element) => {
                        element.innerHTML = element.innerHTML + ("<br />  (game complexity is " + response + ")").italics().fontsize(3).fontcolor("green");
                        },
                     temp);
            
                 
            }
        }
    }
}, 1000)




/*  
your code of content script
goes here
*/



}
}
chrome.runtime.onMessage.addListener(receivefunc);