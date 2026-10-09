-- JARVIS.app: a stay-open applet. While it is open (dot under the Dock icon), JARVIS runs.
--   open the app      → starts JARVIS and opens the browser
--   click the icon    → opens the browser again
--   Quit (⌘Q / Dock)  → turns JARVIS off
-- The real work is in launcher.sh; build the app with launcher/build-app.sh.

property launcher : ""
property serverURL : ""
property ownsServer : false

on run
	set launcher to quoted form of (POSIX path of (path to me) & "Contents/Resources/launcher.sh")
	set serverURL to do shell script launcher & " url"
	try
		do shell script launcher & " start"
		set ownsServer to true
		openInterface()
	on error message number code
		if code is 11 then -- already running (started by JARVIS.command)
			openInterface()
		else if code is 13 then -- first run or an update: install in the Terminal
			set command to POSIX path of (path to me) & "../JARVIS.command"
			do shell script "open -a Terminal " & quoted form of command
		else if code is 10 then
			display alert "Não achei o projeto." message "Deixe o JARVIS.app dentro da pasta agenteIA. Para o Dock, arraste o app até lá (o Dock guarda só um atalho)."
		else if code is 12 then
			display alert "A porta está ocupada." message "Outro programa usa " & serverURL & ". Feche-o e abra o JARVIS de novo."
		else
			showFailure("O JARVIS não conseguiu iniciar.")
		end if
		quit
	end try
end run

on reopen
	if ownsServer then openInterface()
end reopen

on idle
	if ownsServer then
		try
			do shell script launcher & " alive"
		on error
			set ownsServer to false
			showFailure("O JARVIS parou.")
			quit
		end try
	end if
	return 10
end idle

on quit
	if ownsServer then
		try
			do shell script launcher & " stop"
		end try
		set ownsServer to false
	end if
	continue quit
end quit

on openInterface()
	if (system attribute "JARVIS_NO_BROWSER") is "" then open location serverURL
end openInterface

on showFailure(title)
	set answer to display alert title message "Os detalhes estão em data/logs/jarvis.log." buttons {"Fechar", "Ver log"} default button "Ver log"
	if button returned of answer is "Ver log" then
		do shell script "open -e " & quoted form of (POSIX path of (path to me) & "../data/logs/jarvis.log")
	end if
end showFailure
