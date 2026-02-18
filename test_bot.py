from app.chat.bot import TripChatBot
from aegis.config import AEGISConfig

print("Initializing bot...")
bot = TripChatBot(AEGISConfig())
print(f"Bot initialized. trip_data has {len(bot.trip_data['days'])} days")
print(f"First day has {len(bot.trip_data['days'][0]['plan'])} activities")
if bot.trip_data['days'][0]['plan']:
    print(f"First activity: {bot.trip_data['days'][0]['plan'][0]['location']}")
