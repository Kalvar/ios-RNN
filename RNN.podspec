Pod::Spec.new do |s|
  s.name         = "RNN"
  s.version      = "1.0"
  s.summary      = "Simple Recurrent Neural Network."
  s.description  = <<-DESC
                   Simple Recurrent Neural Network that familiar with time series analysis.
                   DESC
  s.homepage     = "https://github.com/Kalvar/ios-RNN"
  s.license      = { :type => 'MIT', :file => 'LICENSE' }
  s.author       = { "Kalvar Lin" => "ilovekalvar@gmail.com" }
  s.social_media_url = "https://twitter.com/ilovekalvar"
  s.source       = { :git => "https://github.com/Kalvar/ios-RNN.git", :tag => s.version.to_s }
  s.platform     = :ios, '9.0'
  s.requires_arc = true
  s.public_header_files = 'RNN/**/*.h'
  s.source_files = 'RNN/**/*.{h,m}'
  s.frameworks   = 'Foundation'
end
