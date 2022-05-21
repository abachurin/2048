### Running the Bot in AWS

You can find current public IP address at EC2 instance page in AWS. Use **http**, without "s", to access it. The EC2 instance is usually permanently open, so the address doesn't change. If it fails/stops/not working for any reason - you have to retrace the Deplyment steps (described below) from some point, then check the new public IP address.

### Deployment to AWS

It helps to know some basics about Docker. Worth watching some basic tutorial on AWS, about Console, how to create IAM role etc. The specifics of Amazon, screenviews and such, are changing with time, so this very good and clear video will probably become completely outdated one day, but as of 18/07/2021 it is almost all you need:

https://www.youtube.com/watch?v=zs3tyVgiBQQ

I will now describe the main steps.

1. Create AWS account. Copy-paste your account number somewhere, we will need it later. You can find it in your profile, should look like "My account 650019805333" - this is the number we need.

2. Note that to find any AWS service, type the name in "Search" on the top.

3. Find IAM. On the left choose "Users", then on the right "Add users". Fill it like this:

<img width="1023" alt="Screenshot 2021-07-30 at 13 01 53" src="https://user-images.githubusercontent.com/43069886/127650188-f7477fcc-1f62-4230-b313-a8fa1752af63.png">

4. On the next screen choose "Create group" fill as below:

<img width="1441" alt="Screenshot 2021-07-30 at 13 06 15" src="https://user-images.githubusercontent.com/43069886/127650699-34474348-ad4c-49c3-820e-e0cfd187aaee.png">
Then click through next couple of screens and "Create user"

5. If you don't have yet, install Docker https://docs.docker.com/get-docker/ ...

6. ... and AWS CLI. On Mac i just run `pip3 install awscli --upgrade --user` in the Terminal. Instructions for any system: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv1.html

7. In the Terminal (same as command line in Windows, probably) run:
`aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 650019805333.dkr.ecr.eu-west-1.amazonaws.com`
- only first put in your account number from p.1., and your AWS region in 2 places where you see "eu-west_1" (my region), which you can find right next to your profile name in the top menu

8. You should see "Login Succeeded" message after several seconds. If you get any error, as i did at first - well ... took me some time to fix, don't remember what was wrong and how i fixed it exactly, just a lot of googling , trial-and-error and voi la, all works fine :)

9. Find ECR, "Elastic Container Registry", and "Create repository" there, you only need to invent a name, say 'mybot'. Copy the address of the repository you've just created.

10. Now go to the directory with our python project (if you git-cloned it from Github, should end with "Raj_bot"). There run in the Terminal:
`docker-compose build` - wait until it builds, then run
`docker tag raj_bot_dash:latest <repository adrress here>:latest` - then run
`docker push <repository adrress here>:latest` - the process can take some time at first run. When you referesh the image next timel, will be faster.
Now check that the image appeared in AWS repository, you should see it there with 'latest' tag. Copy-save "Image URI" of the image, we will need it later. From this point we only work in AWS.

11. Find ECS, "Elastic Container Service", and to "Clusters" there in the menu on the left. Actually, this particular process is very well described in the video link, with lots of pictures. Start it from 6-15 time. Some comments:
a) While choosing "EC2 instance type" - Free-Tier is currently "t2-micro" (something different in the video).
b) in "Task Definition" don't bother with Task Size (memory and CPU), default is fine.
c) In "Add Container" choose "Hard Limit" 512 for memory.

12. Now the task is running and all looks fine on AWS, but you probably can't access your file. Click in EC2 instance and on the EC2 screen choose "security" -> "security groups" -> "inbound rules" -> "edit inbound rules". Add rules so it looks like below:

<img width="1521" alt="Screenshot 2021-07-30 at 15 49 09" src="https://user-images.githubusercontent.com/43069886/127671018-fba828b6-0786-4928-a651-fe88437d8f54.png">

13. Now at EC2 Instance page you have Public IPv4 DNS, e.g. right now it is ec2-34-245-39-16.eu-west-1.compute.amazonaws.com. Just go there, change to http:// AND add :8888 in the end. (AWS addresses are consiodered unsafe by Mac, maybe no need to change on Windows machine), and you should see this:

<img width="1634" alt="Screenshot 2021-07-30 at 15 55 57" src="https://user-images.githubusercontent.com/43069886/127671980-6b40a4d7-920b-4833-9792-e8715f1d5429.png">

14. Oh, last thing, i added some basic authentification login/password = raj/1234 

### Running the Bot locally

Them easiest way is just to run the executive Dash module `main.py`, it will work just fine (at least on Mac).
The trading module per se is `bot.py`. One can run it directly. The __main__ program there would look like this:
```
b = Bot(model_params={}, mode_of_action='sim', use_real_cash=False, fresh_start=False)
b.session(open_any_time=False)
```
Models params like `stop_loss`, `expiration` etc. can be changed in `bot_parameters.json` file. If it doesn't find some params there - will take from default_params at the top of the module.

If `fresh_start=True` : the Bot deletes all previous data, cancels all working orders, liquidates all open positions, then starts anew.

If `use_real_cash=True` : the Bot takes real cash on account, disregarding `start_capital` parameter.

Finally, if `open_any_time=True` : the Bot starts buying top-picks when launched and generally ignores the real timing. This option is mostly for testing/debugging. Otherwise, when `open_any_time=False` , the Bot does thing according to real market schedule. Before open in collects previous day's data, calculates top-p;icks and how much cash should be allocated to them, resends stop-profit orders etc. Then it send BUY orders for top-picks on the open. Then monitores working orders regularly (parameter `check_execution` = 60 seconds by default). At pre-close time, which i set 15 minutes before close, it cancels stop-profit orders for stocks that have expiration today and sells them. Then the Bot reconciles internal positions with real ones, saves all data, sends logs and trqades .csv file to a recipients list, set in `mail.py` module, and finally exits.

There are two different functions in then Bot class both for market open and close, ones with the MOO/MOC orders, the other without. In the real Market it is preferable to use "auction"-based functions. Simulated modes sometimes don't allow them (example - Tradestation).

### Parameters for Tradestation API connections in the file necessary_ts_info.txt

Take a look at this file. First two lines with `refresh_tokens` are filled by the Bot at some time. Others need to be filled in the same format.

### Mail.py

There are 3 parameters need to be filled:
```
my_mail_address = 'mail of the server where the bot is running'
password_for_mail = 'password fot that mail'
bot_info_recipients = ['bachurin.alex@gmail.com', 'rajsharmaofficial@gmail.com']
```
If the Bot is unable to send the mail - it writes the relevant warning to the log file and keeps working.

### Engine_ts.py

Contains all functions/methods necessary for Tradestation API connection. It all works automatically unless you need a new "Refresh Token", e.g. when you get new API keys, or want to start working in a new mode (say, 'live' versus 'sim'). Getting new Refresh Token is a bit "captcha"-style, so has to be done manually. But very easy! Just put in login/password and answer the secret question, the received token will then be put into the relevant .txt file by the Bot. Then restart the Bot. The Refresh Token, once obtained for the given API keys/mode, is valid indefinitely.

### How to add another engine for another Platform

1. Write an analog of `engine_ts.py` file. I has to contain the `Client` class, responcible for API connection. Then add import from that file on top of `bot.py` and make the relevant change in the relevant line in Bot: `self.trading_api = Client_TS(sim=self.mode_of_action)`.

2.  This `Client` class has to contain the following end-functions used by the Bot:
```
get_positions()
get_account_balance()
get_orders()
submit_limit_order()
submit_market_order()
submit_oco_order()
cancel_order()
```
Those functions need to have the same signature as in `engine_ts.py`
