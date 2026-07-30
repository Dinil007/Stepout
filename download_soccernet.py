from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory="SoccerNet")
downloader.password = "s0cc3rn3t"

print("Starting download...")

downloader.downloadGames(
    split=["train"],
    files=["1_720p.mkv", "2_720p.mkv"],
    verbose=True
)

print("Finished")