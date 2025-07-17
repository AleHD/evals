source .env

while true; do
	date
	uv run scripts/automate.py
	echo a mimir...
	sleep $(( 60*60 ))
done
