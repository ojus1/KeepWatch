# KeepWatch - Malicious URL Detection using Machine Learning

This is a hobby project, a browser extension that will analyze URLs and server related metadata such as Whois, Location (IP address), Reputation etc, these are sent to a Machine learning model embedded in the browser extension. If the model classifies the URL as Malicious, the extension will throw a popup warning the user.

![Screen Shot](./example.png?raw=true)

# Confusion Matrix for the Decision tree classifier
![Confusion Matrix](./DecisionTreeCNF.png?raw=true)

### Prerequisites

Google Chrome


### Installation

Clone this repository.
Go to [chrome://extensions](chrome://extensions)
Click on "Load Unpacked"
Load the "KeepWatch" folder
The extension should be up and running.


## Authors

* **Surya Kant Sahu** - *Initial work* - [ojus1](https://github.com/ojus1)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Packages used for Machine learning model: Python: Pandas, Sklearn, Sklearn-Porter
* Used "Natural"-NPM module's RegExpTokenizer source code for reference