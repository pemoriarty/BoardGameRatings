
chrome.tabs.query({'active': true, 'lastFocusedWindow': true}, function (tabs) {
    var url = tabs[0].url;
    console.log(url);
    document.getElementById("myText").innerHTML = url;

});



document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('checkPage').addEventListener('click', function(){
      chrome.tabs.query({'active': true, 'lastFocusedWindow': true}, function (tabs) {
        
     
   
    
    
         
      });
});
});