
function sendfunc(tab){
msg={txtt:"execute"};
chrome.tabs.sendMessage(tab.id,msg);
}
chrome.browserAction.onClicked.addListener(sendfunc);