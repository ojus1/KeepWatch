var utils = require("./utils");
var natural = require("natural");

var charLenLim = 14;

extract_features("adserving.favorit-network.com/eas?camp=19320;cre=mu&grpid=1738&tag_id=618&nums=FGApbjFAAA");

function extract_features(text){
    let tokenizer = new natural.RegexpTokenizer({pattern: /((\d+)|(\W+)|(_)+)+/});
    tokenized = tokenizer.tokenize(text);
    tokenized = [...new Set(tokenized)];
    tokenized = tokenized.filter(item => item !== undefined);
    console.log(tokenized.sort());
}